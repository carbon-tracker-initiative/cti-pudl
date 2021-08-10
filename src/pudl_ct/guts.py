"""Prepare unit level data from EIA, FERC1, NEMS and CEMS for use with GEM."""
# Standard libraries
import logging
import pathlib
from collections import Counter, defaultdict
import itertools
from ast import literal_eval

from copy import deepcopy
import dask.dataframe as dd
import numpy as np
import pandas as pd
import warnings

import pudl

logger = logging.getLogger(__name__)

################################
# global variables
################################
IDX_GEN = ['report_date', 'plant_id_eia', 'generator_id']
IDX_UNIT = ['report_date', 'plant_id_eia', 'unit_id_pudl']
idx_unit_eia = ['report_year', 'plant_id_eia', 'unit_id_eia']
idx_plant_fuel = ['report_date', 'plant_id_pudl', 'fuel_type_code_pudl']
IDX_PLANT_FUEL = ['report_date', 'plant_id_pudl', 'fuel_type_code_pudl']
IDX_PLANT_PUDL = ['report_date', 'plant_id_pudl', ]
IDX_BOILER = ['plant_id_eia', 'report_date', 'boiler_id']
# IDX_STEAM = ['utility_id_ferc1', 'plant_name_ferc1', 'report_year']
IDX_STEAM = ['utility_id_ferc1', 'plant_id_ferc1', 'report_year']


COLS_CAPEX = ['capex_total_shifted', 'capex_annual_addt',
              'capex_annual_addt_rolling']

COLS_SUM_CEMS = [
    'so2_mass_lbs',
    'nox_mass_lbs',
    'co2_mass_tons',
    'operating_time_hours',
    'gross_load_mw'
]

WA_COL_DICT_STEAM = {
    'opex_nonfuel_per_mwh': 'net_generation_mwh',
    'opex_fuel_per_mwh': 'net_generation_mwh',
    'capex_per_mw': 'capacity_mw',
    'capex_annual_per_mwh': 'net_generation_mwh',
    'capex_annual_per_mw': 'capacity_mw',
    'capex_annual_per_kw': 'capacity_mw',
    'capex_annual_per_mwh_rolling': 'net_generation_mwh',
    'capex_annual_per_mw_rolling': 'capacity_mw',
}


PATH_INPUTS = pathlib.Path().cwd().parent / 'inputs'
path_nems = PATH_INPUTS / 'pltf860.csv'
path_nems_infl = PATH_INPUTS / 'NEMS_GDP_infl.xlsx'

NEMS_FILE_NAMES = {
    2017: 'pltf860.v1.201.txt',
    2018: 'pltf860.v1.206.txt',
    2019: 'pltf860.v1.210.txt',
}

PATH_GEM = PATH_INPUTS / 'gem_gas_units.xlsx'

MUL_COLS = [
    'record_id_eia', 'plant_name_new', 'plant_part', 'report_year',
    'ownership', 'plant_name_eia', 'plant_id_eia', 'generator_id',
    'unit_id_pudl', 'prime_mover_code', 'energy_source_code_1',
    'technology_description', 'ferc_acct_name', 'utility_id_eia',
    'utility_id_pudl', 'true_gran', 'appro_part_label', 'appro_record_id_eia',
    'record_count', 'fraction_owned', 'ownership_dupe'
]


def make_ct_compilation(pudl_out, unit_id_col):
    """
    Make CT's plant-unit output with either PUDL or GEM unit ID.

    Args:
        pudl_out
        unit_id_col (str): either unit_id_pudl or unit_id_gem, depending on
            which unit you'd like the output to be aggregated by.
    """
    steam_df = prep_plants_ferc(pudl_out).pipe(calc_annual_capital_addts_ferc1)

    # grab the generators & aggregate to units
    gen = (
        prep_gens_eia(pudl_out)
        .pipe(add_nems, pudl_out)
        .pipe(merge_gem_w_df, df_source='eia')
    )

    unit = agg_gen_eia_to_unit(gen=gen, unit_id_col=unit_id_col)

    gens_w_ferc1 = (
        merge_eia_ferc(
            gen=gen,
            steam_df=steam_df,
            unit_id_col=unit_id_col,
            unit=unit
        )
        .pipe(fill_in_opex_w_nems)
        .pipe(add_lifetime_avg_capex, unit_id_col)
        .pipe(add_state_fuel_avg)
    )

    if unit_id_col == 'unit_id_gem':
        gens_w_ferc1 = merge_onto_gem(gens_w_ferc1)
    # replace the 0's with nulls
    gens_w_ferc1.net_generation_mwh.mask(
        gens_w_ferc1.net_generation_mwh == 0, pd.NA, inplace=True)

    return gens_w_ferc1


def merge_onto_gem(gens_w_ferc1):
    """
    Merge the CT compilation onto GEM (for the GEM plant & unit name).

    Args:
        gens_w_ferc1 (pandas.DataFrame):
    """
    out = pd.merge(
        get_gem(),
        gens_w_ferc1,
        on=['plant_id_eia', 'unit_id_gem'],
        validate='1:m',
        how='left'
    )
    return out


def add_state_fuel_avg(out_df):
    """Add state wide fuel cost averages."""
    idx_fuel_state = ['fuel_type_code_pudl', 'report_date', 'state']
    state_fuel_avg = weighted_average(
        out_df,
        {'fuel_cost_per_mmbtu': 'net_generation_mwh'},
        idx_fuel_state
    )
    out_df_w_avg = (
        out_df.merge(
            state_fuel_avg,
            how='left',
            on=idx_fuel_state,
            validate='m:1',
            suffixes=('', '_state_avg')
        )
        .assign(
            fuel_cost_source=lambda x:
                np.where(
                    x.fuel_cost_per_mmbtu.notnull(),
                    'plant-level',
                    np.where(
                        x.fuel_cost_per_mmbtu_state_avg.notnull(),
                        'state-level', pd.NA
                    )
                ),
            fuel_cost_per_mmbtu=lambda x:
                x.fuel_cost_per_mmbtu.fillna(x.fuel_cost_per_mmbtu_state_avg)
        )
    )
    return out_df_w_avg

################################
# EIA Data Prep
################################


def prep_gens_eia(pudl_out):
    """
    Prepate a table with EIA generator data.

    Use the generators EIA860 table as the 'backbone' (it should have all
    of the generators ids). Merge in the MCOE table for additional
    calcualted values and the boiler generator associations for the unit id
    and the boiler id (for merging with CEMS data). This function also
    generates a 'composite unit-gen id'.

    Args:
        pudl_out

    Returns:
        pandas.DataFrame

    """
    # prep for the gens/mcoe merge... they have a ton of overlapping columns.
    # we want to use the gens columns, so we'll drop the non-PK columns
    # from mcoe
    drop_cols = [
        x for x in pudl_out.gens_eia860()
        if x in pudl_out.mcoe() and x not in IDX_GEN
    ]

    gen = (
        pd.merge(
            pudl_out.gens_eia860(),
            pudl_out.mcoe().drop(columns=drop_cols),
            on=IDX_GEN,
            how='outer',
            validate='1:1',
        )
        .merge(
            pudl_out.bga()[['plant_id_eia', 'report_date',
                            'generator_id', 'unit_id_eia']].drop_duplicates(),
            on=['plant_id_eia', 'report_date', 'generator_id'],
            how='outer',
            validate='1:1'
        )
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        .assign(
            composite_unit_gen_id=composite_id_assign,
            report_year=lambda x: x.report_date.dt.year
        )
        # add the ferc_acct
        .merge(
            pd.read_csv(
                pathlib.Path().cwd().parent / 'inputs/ferc_acct_map.csv')
            [['technology_description', 'prime_mover_code', 'ferc_acct_name']]
            .drop_duplicates()
            .dropna(),
            how='left'
        )
        .drop_duplicates(subset=IDX_GEN)

    )
    return gen


def _prep_gen_agg_dict():
    """Generate aggregation dict with cols and agg methods."""
    cols_sum = ['capacity_mw', 'net_generation_mwh', 'capacity_mw_nems',
                'net_generation_mwh_nems', 'variable_om', 'fixed_om']
    cols_str = ['generator_id', 'unit_id_eia', 'fuel_type_code_pudl',
                'energy_source_code_1', 'sector_name', 'operational_status']
    cols_max_date = ['planned_retirement_date', 'retirement_date']
    cols_min_date = ['operating_date']

    agg_dict = {
        **{k: str_squish for k in cols_str},
        **{k: sum for k in cols_sum},
        **{k: max for k in cols_max_date},
        **{k: min for k in cols_min_date},
    }
    return agg_dict


def agg_gen_eia_to_unit(gen, unit_id_col):
    """
    Aggregate EIA generator data.

    Via both weithed averages and standard groupby.agg methods.

    Args:
        gen (pandas.DataFrame): generator data (result of prep_gens_eia)
    """
    logging.info(
        "Aggergating the generators to a unit level. This takes some minutes."
    )
    idx_unit = ['plant_id_eia', 'report_date', unit_id_col]
    # prep the categorical data aggregation
    # we want to prioritize some
    gen = gen.astype({"operational_status": pd.CategoricalDtype(), })
    gen['operational_status'].cat.set_categories(
        ['existing', 'proposed', 'retired', ], inplace=True)
    gen = gen.sort_values('operational_status')

    agg_dict = _prep_gen_agg_dict()
    gen_gpb = gen.groupby(idx_unit, dropna=False).agg(
        agg_dict, min_count=1).reset_index()

    cols_wt_avg_net_gen = [
        'heat_rate_mmbtu_mwh', 'capacity_factor', 'capacity_factor_nems',
        'fuel_cost_per_mmbtu', 'fuel_cost_per_mwh', 'fixed_om_kw',
        'fixed_om_mwh', 'variable_om_mwh', 'fix_var_om_mwh',
        'fixed_v_total_ratio', 'var_v_total_ratio',
    ]
    wt_avg_dict = {k: 'net_generation_mwh' for k in cols_wt_avg_net_gen}

    cols_wt_avg_cap = ['capex_annual_per_kw_nems']
    wt_avg_dict.update({k: 'capacity_mw' for k in cols_wt_avg_cap})
    gen_wtavg = weighted_average(
        gen, wt_avg_dict, idx_unit)

    idx_plant = ['plant_id_eia', 'report_date']
    gen_agg = (
        pd.merge(
            gen_gpb,
            gen_wtavg,
            on=idx_unit,
            how='outer',
            validate='1:1'
        )
        .merge(
            gen[idx_plant + [
                'plant_id_pudl', 'plant_name_eia', 'latitude',
                'longitude', 'city', 'county', 'state', 'utility_id_eia',
                'utility_id_pudl', 'utility_name_eia']]
            # remove subset=idx_plant when this issue is closed
            # https://github.com/catalyst-cooperative/pudl/issues/1033
            .drop_duplicates(subset=idx_plant),
            how='outer',
            on=idx_plant,
            validate="m:1"
        )
    )
    return gen_agg

################################
# Simple FERC1
################################


def prep_plants_ferc(pudl_out):
    """
    Prep steam plants table from FERC1.

    Add fuel type per records via `pudl_out.fbp_ferc1`
    """
    fpb_cols_to_use = [
        'report_year', 'utility_id_ferc1', 'plant_name_ferc1',
        'utility_id_pudl', 'fuel_cost', 'fuel_mmbtu',
        'primary_fuel_by_mmbtu',
    ]

    steam_df = (
        pd.merge(
            pudl_out.plants_steam_ferc1(),
            pudl_out.fbp_ferc1()[fpb_cols_to_use],
            on=['report_year',
                'utility_id_ferc1',
                'utility_id_pudl',
                'plant_name_ferc1',
                ],
            how='left')
        .assign(
            report_date=lambda x: pd.to_datetime(x.report_year, format='%Y')
        )
        .rename(columns={'primary_fuel_by_mmbtu': 'fuel_type_code_pudl'})
    )
    return steam_df


def calc_annual_capital_addts_ferc1(steam_df, window=3):
    """
    Calculate annual capital additions for FERC1 steam records.

    Convert the capex_total column into annual capital additons the
    `capex_total` column is the cumulative capital poured into the plant over
    time. This function takes the annual difference should generate the annual
    capial additions. It also want generates a rolling average, to smooth out
    the big annual fluxuations.

    Args:
        steam_df (pandas.DataFrame): result of `prep_plants_ferc()`

    Returns:
        pandas.DataFrame: augemented version of steam_df with two additional
        columns: `capex_annual_addt` and `capex_annual_addt_rolling`.
    """
    # we need to sort the df so it lines up w/ the groupby
    steam_df = steam_df.sort_values(IDX_STEAM)
    # we group on everything but the year so the groups are multi-year unique
    # plants the shift happens within these multi-year plant groups
    steam_df['capex_total_shifted'] = steam_df.groupby(
        [x for x in IDX_STEAM if x != 'report_year'])[['capex_total']].shift()
    steam_df = steam_df.assign(
        capex_annual_addt=lambda x: x.capex_total - x.capex_total_shifted
    )

    addts = pudl.helpers.generate_rolling_avg(
        steam_df,
        group_cols=[x for x in IDX_STEAM if x != 'report_year'],
        data_col='capex_annual_addt',
        window=window
    )

    steam_df_w_addts = (
        pd.merge(
            steam_df,
            addts[IDX_STEAM + ['capex_total', 'capex_annual_addt_rolling']],
            on=IDX_STEAM + ['capex_total'],
            how='left',
        )
        .assign(
            capex_annual_per_mwh=lambda x:
                x.capex_annual_addt / x.net_generation_mwh,
            capex_annual_per_mw=lambda x:
                x.capex_annual_addt / x.capacity_mw,
            capex_annual_per_kw=lambda x:
                x.capex_annual_addt / x.capacity_mw / 1000,
            capex_annual_per_mwh_rolling=lambda x:
                x.capex_annual_addt_rolling / x.net_generation_mwh,
            capex_annual_per_mw_rolling=lambda x:
                x.capex_annual_addt_rolling / x.capacity_mw,
        )
    )

    steam_df_w_addts = add_mean_cap_addts(steam_df_w_addts)
    # bb tests for volumne of negative annual capex
    neg_cap_addts = len(
        steam_df_w_addts[steam_df_w_addts.capex_annual_addt_rolling < 0]) \
        / len(steam_df_w_addts)
    neg_cap_addts_mw = (
        steam_df_w_addts[
            steam_df_w_addts.capex_annual_addt_rolling < 0]
        .net_generation_mwh.sum()
        / steam_df_w_addts.net_generation_mwh.sum())
    message = (f'{neg_cap_addts:.02%} records have negative capitial additions'
               f': {neg_cap_addts_mw:.02%} of capacity')
    if neg_cap_addts > .1:
        warnings.warn(message)
    else:
        logger.info(message)
    return steam_df_w_addts


def add_lifetime_avg_capex(steam_df, unit_id_col):
    """Add lifetime average of capital additions."""
    idx_unit_non_annual = ['plant_id_eia', unit_id_col]

    unit_life_gb = steam_df.groupby(idx_unit_non_annual, as_index=False)

    steam_df_w_avg_life = (
        pd.merge(
            steam_df,
            unit_life_gb[['capex_annual_per_kw']].mean(),
            on=idx_unit_non_annual,
            how='left',
            suffixes=('', '_avg_life')
        )
    )
    return steam_df_w_avg_life


def add_mean_cap_addts(steam_df):
    """Add mean capital additions over lifetime of plant (via `IDX_STEAM`)."""
    idx_steam_no_date = [c for c in IDX_STEAM if c != 'report_year']
    gb_cap_an = steam_df.groupby(idx_steam_no_date)[['capex_annual_addt']]
    # calcuate the standard deviatoin of each generator's capex over time
    df = (
        steam_df
        .merge(
            gb_cap_an.std().add_suffix('_gen_std').reset_index(),
            how='left',
            on=idx_steam_no_date,
            validate='m:1'  # should this really be 1:1?
        )
        .merge(
            gb_cap_an.mean().add_suffix('_gen_mean').reset_index(),
            how='left',
            on=idx_steam_no_date,
            validate='m:1'  # should this really be 1:1?
        )
        .assign(
            capex_annual_addt_diff_mean=lambda x: x.capex_annual_addt - \
            x. capex_annual_addt_gen_mean,
        )
    )
    return df


def agg_plants_ferc_by_plant_fuel(steam_df):
    """Aggregate FERC steam plants by plant fuel."""
    steam_by_fuel = (
        pd.merge(
            weighted_average(
                steam_df,
                wa_col_dict=WA_COL_DICT_STEAM,
                idx_cols=idx_plant_fuel
            ),
            (
                steam_df.groupby(idx_plant_fuel, as_index=False)
                [['opex_nonfuel', 'net_generation_mwh', 'capex_total']
                 + COLS_CAPEX]
                .sum(min_count=1)
            ),
            on=idx_plant_fuel,
            how='outer',
            validate='1:1'
        )

        .merge(  # Is this doing nothing??? can I remove it??
            steam_df.groupby(IDX_PLANT_PUDL, as_index=False)
            [[]].sum(min_count=1),
            on=IDX_PLANT_PUDL,
            how='outer',

        )
        .merge(
            weighted_average(
                steam_df,
                wa_col_dict=WA_COL_DICT_STEAM,
                idx_cols=IDX_PLANT_PUDL),
            on=IDX_PLANT_PUDL,
            how='outer',
            suffixes=('', '_plant')
        )
    )
    return steam_by_fuel


def merge_eia_ferc_simple(unit, steam_df, unit_id_col):
    """Merge EIA and FERC via plant-fuel aggregations."""
    # aggregate the steam table by plant-fuel
    steam_by_fuel = agg_plants_ferc_by_plant_fuel(steam_df)

    gens_w_ferc1 = (
        pd.merge(
            unit,
            pudl.helpers.convert_cols_dtypes(steam_by_fuel, 'eia'),
            on=idx_plant_fuel,
            how='outer',
            indicator=True,
            validate='m:1',
            suffixes=('', '_ferc1')
        )
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        .assign(
            opex_nonfuel_per_mwh=lambda x:
                x.opex_nonfuel_per_mwh.fillna(x.opex_nonfuel_per_mwh_plant)
        )
    )
    gens_w_ferc1 = (
        gens_w_ferc1.loc[gens_w_ferc1._merge != 'right_only']
        .drop(columns=['_merge', 'opex_nonfuel_per_mwh_plant'])
        .assign(
            opex_fixed_per_mwh=lambda x: np.where(
                (x.fuel_type_code_pudl.notnull() &
                 (x.fuel_type_code_pudl == 'coal')),
                x.opex_nonfuel_per_mwh * .811,
                pd.NA),
            opex_variable_per_mwh=lambda x: np.where(
                (x.fuel_type_code_pudl.notnull() &
                 (x.fuel_type_code_pudl == 'coal')),
                x.opex_nonfuel_per_mwh * .188,
                pd.NA),
            steam_agg_type="plant-fuel"
        )
    )
    return gens_w_ferc1


################################
# Granual FERC1 Connection
################################

def grab_ferc1_to_eia_connection():
    """
    Get the FERC/EIA connection.

    Args:
        file_path_ferc1_to_eia (path-like): path to pickled table
    """
    file_path_ferc1_to_eia = PATH_INPUTS / 'ferc1_to_eia.pkl.gz'
    ferc1_to_eia = (
        pd.read_pickle(file_path_ferc1_to_eia)[['record_id_ferc1'] + MUL_COLS]
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        .dropna(subset=['record_id_eia'])
        .assign(composite_unit_gen_id=composite_id_assign)
    )
    return ferc1_to_eia


def prep_ferc1_to_eia(ferc1_to_eia, steam_df):
    """Prepate the."""
    steam = (
        pd.merge(
            steam_df.rename(columns={'record_id': 'record_id_ferc1'}),
            ferc1_to_eia,
            on=['record_id_ferc1'],
            how='outer',
            suffixes=('', '_eia'),
            validate='1:1'
        )
        # .assign(plant_part_eia='plant_unit',)
        # report_date=lambda x: pd.to_datetime(x.report_year, format='%Y')
        .rename(columns={'plant_part': 'plant_part_ferc1'})
    )
    return steam


def count_unique_ids_per_plant_part(steam, gen, id_col='unit_id_pudl'):
    """Count unique unit_id's records per plant_part."""
    steam_w_count = deepcopy(steam).astype(
        {'plant_id_eia': pd.Int64Dtype()}).dropna(subset=['plant_id_eia'])
    # grab the dictionary which contains info about the plant_parts
    # in the master unit list.... this will help us know the names and
    # indentifying columns for each plant_part within the EIA master
    # unit list
    plant_part_idx = {
        'plant': [
            'plant_id_eia', 'report_date'],
        'plant_gen': [
            'plant_id_eia', 'generator_id', 'report_date'],
        'plant_technology': [
            'plant_id_eia', 'technology_description', 'report_date'],
        'plant_prime_fuel': [
            'plant_id_eia', 'energy_source_code_1', 'report_date'],
        'plant_prime_mover': [
            'plant_id_eia', 'prime_mover_code', 'report_date'],
        'plant_ferc_acct': [
            'plant_id_eia', 'ferc_acct_name', 'report_date']
    }
    for part_name, idx_part in plant_part_idx.items():
        part_df = steam.loc[steam.plant_part_ferc1 == part_name][idx_part]
        logger.debug(f'preparing count for {part_name}')
        logger.debug(f'idx columns: {idx_part}')
        part_unit = (
            pd.merge(
                part_df,
                gen[idx_part + [id_col]],
                on=idx_part
            )
        )
        part_unit_count = (
            part_unit
            .groupby(idx_part)
            .nunique(dropna=False)
            .add_suffix('_count')
        )
        part_unit_id = (
            part_unit
            .groupby(idx_part)
            .agg({id_col: str_squish})
            .add_suffix('_agg')
        )

        steam_w_count = steam_w_count.set_index(idx_part)
        steam_w_count.loc[steam_w_count.plant_part_ferc1 ==
                          part_name, f'{id_col}_count'] = part_unit_count
        steam_w_count.loc[steam_w_count.plant_part_ferc1 ==
                          part_name, f'{id_col}_agg'] = part_unit_id
        steam_w_count = steam_w_count.reset_index()

    steam_w_count.loc[steam_w_count.plant_part_ferc1 ==
                      'plant_unit', f'{id_col}_count'] = 1
    if not steam_w_count[
        (steam_w_count[f'{id_col}_count'] == 1)
            & steam_w_count[f'{id_col}_agg'].str.contains(';')].empty:
        raise AssertionError(
            f"Records w/ one {id_col} should never have a aggregated unit id."
        )
    return steam_w_count


def count_unique_steam_records_per_id(steam_w_count, unit_id_col):
    """
    Count unique steam plant records per unit id.

    Args:
        steam_w_count (pandas.DataFrame): result of
            `count_unique_ids_per_plant_part`
    """
    idx_unit = ['report_date', 'plant_id_eia', unit_id_col]
    steam_w_count = (
        pd.merge(
            steam_w_count,
            steam_w_count.assign(idx_unit_count=1)
            .groupby(idx_unit + [f'{unit_id_col}_count'], dropna=False)
            [['idx_unit_count']].count().reset_index(),
            on=idx_unit + [f'{unit_id_col}_count'],
            how='left',
            validate='m:1'
        )
    )
    return steam_w_count


def agg_one_id_steam(steam_w_count, unit_id_col):
    """
    Find the unit compatible steam records.

    Args:
        steam_w_count (pandas.DataFrame)
    """
    idx_unit = ['report_date', 'plant_id_eia', unit_id_col]
    steam_1_1 = steam_w_count.loc[
        (steam_w_count[f'{unit_id_col}_count'] == 1)
        & (steam_w_count.idx_unit_count == 1)]
    steam_1_m = steam_w_count.loc[
        (steam_w_count[f'{unit_id_col}_count'] == 1)
        & (steam_w_count.idx_unit_count != 1)
    ]
    steam_1_m.loc[:, unit_id_col] = steam_1_m.loc[:, f'{unit_id_col}_agg']

    logger.info(
        f"{(len(steam_1_1) + len(steam_1_m)) / len(steam_w_count):.02%} "
        "of steam records match with one unit"
    )
    cols_ferc1 = (
        list(WA_COL_DICT_STEAM.keys())
        + ['opex_nonfuel', 'net_generation_mwh', 'capex_total']
        + COLS_CAPEX)
    # ['opex_nonfuel_per_mwh', 'capex_per_mw']
    steam_1_1_agg = steam_1_1.loc[:, idx_unit +
                                  cols_ferc1].assign(steam_agg_type="1:1")
    steam_1_m_wtg = weighted_average(steam_1_m.copy(),  WA_COL_DICT_STEAM,
                                     idx_cols=idx_unit)
    steam_1_m_sum = (
        steam_1_m.groupby(idx_unit, as_index=False)
        [['opex_nonfuel', 'net_generation_mwh',
          'capex_total'] + COLS_CAPEX].sum(min_count=1))
    steam_1_m_agg = (
        steam_1_m_wtg
        .merge(
            steam_1_m_sum,
            on=idx_unit,
            how='outer'
        )
        .assign(steam_agg_type="1:m")
    )
    steam_agg = pd.concat([steam_1_1_agg, steam_1_m_agg])
    return steam_agg


def merge_eia_ferc_unit(gen, steam_df, unit_id_col='unit_id_pudl'):
    """Merge EIA and FERC based on."""
    ferc1_to_eia = grab_ferc1_to_eia_connection()
    steam = prep_ferc1_to_eia(ferc1_to_eia, steam_df).pipe(
        merge_gem_w_df, 'ferc')
    eia_ferc_unit = (
        count_unique_ids_per_plant_part(
            steam, gen, id_col=unit_id_col)
        .pipe(count_unique_steam_records_per_id, unit_id_col=unit_id_col)
        .pipe(agg_one_id_steam, unit_id_col=unit_id_col)
        .merge(
            # add in the plant_id_pudl
            gen.loc[:, ['plant_id_pudl', 'plant_id_eia']
                    ].drop_duplicates().dropna(),
            how='left',
            validate='m:1',
        )
    )
    return eia_ferc_unit

################################
# FERC/EIA Connection
################################


def merge_eia_ferc(gen, steam_df, unit_id_col, unit=None, debug=False):
    """Merge EIA and FERC on unit id or by plant-fuel."""
    # create unit records - use these to merge the FERC plant-fuel data onto
    # and
    if unit is None:
        unit = agg_gen_eia_to_unit(gen=gen, unit_id_col=unit_id_col)
    eia_ferc_fuel = merge_eia_ferc_simple(
        unit=unit, steam_df=steam_df, unit_id_col=unit_id_col)
    eia_ferc_unit = merge_eia_ferc_unit(gen, steam_df, unit_id_col)

    idx_unit = ['report_date', 'plant_id_eia', unit_id_col]
    ferc_merge = (
        pd.merge(
            eia_ferc_fuel.loc[:, list(eia_ferc_unit.columns) +
                              ['fuel_type_code_pudl']],
            eia_ferc_unit,
            on=idx_unit + ['plant_id_pudl'],
            how='outer',
            suffixes=('_plant_fuel', '_unit')
        )
    )
    assign_cols = [
        'opex_nonfuel_per_mwh',
        'capex_annual_per_mwh',
        'capex_annual_per_mw',
        'capex_annual_per_kw',
        'capex_annual_addt',
        'capex_annual_addt_rolling',
    ]
    for col in assign_cols:
        ferc_merge.loc[:, col] = ferc_merge[f'{col}_unit'].fillna(
            ferc_merge[f'{col}_plant_fuel'])
    ferc_merge = (
        ferc_merge.assign(
            ferc1_source_level=lambda x: np.where(
                x.opex_nonfuel_per_mwh_unit.notnull(), 'unit',
                np.where(
                    x.opex_nonfuel_per_mwh_plant_fuel.notnull(),
                    'plant_fuel', pd.NA)),
        )
        .pipe(label_multi_method_assoc)
    )
    _ = _check_merge_eia_ferc(ferc_merge)
    if not debug:
        # once we've run the checks, we can drop these fuel/unit columns
        ferc_merge = (
            ferc_merge.drop(columns=(
                ferc_merge.filter(like='_unit')
                + ferc_merge.filter(like='_plant_fuel')).columns
            )
        )
    unit_w_ferc = (
        pd.merge(
            unit,
            ferc_merge,
            on=idx_unit + ['plant_id_pudl'],
            how='left',
            suffixes=('', '_ferc')
        )
        .assign(
            capex_source=lambda x: np.where(
                x.capex_annual_per_kw.notnull(),
                'ferc1', np.where(
                    x.capex_annual_per_kw_nems.notnull(), 'nems', pd.NA)),
            capex_annual_per_kw=lambda x:
                x.capex_annual_per_kw.fillna(x.capex_annual_per_kw_nems)
        )
    )
    capex = unit_w_ferc.loc[unit_w_ferc.capex_annual_per_kw.notnull()]
    cap_percent = capex.capacity_mw.sum() / unit_w_ferc.capacity_mw.sum()
    rec_percent = len(capex) / len(unit_w_ferc)
    logger.info(
        f"We have {cap_percent:.0%} of capacity assocaited with annual capex, "
        f"which is {rec_percent:.0%} of the total records."
    )
    return unit_w_ferc


def label_multi_method_assoc(unit_w_ferc):
    """
    Label the plants that have multiple FERC/EIA connection methods.

    If a part of a plant
    """
    assoc_types = (
        unit_w_ferc
        .groupby(IDX_PLANT_PUDL, dropna=False)
        [['ferc1_source_level']]
        .nunique()
        .reset_index()
        .assign(
            eia_ferc_merge_multi_method_plant=lambda x: np.where(
                x.ferc1_source_level > 1,
                True, False)
        )
        .drop(columns=['ferc1_source_level'])
    )

    gens_w_ferc1_w_label = (
        unit_w_ferc.merge(
            assoc_types,
            on=IDX_PLANT_PUDL
        )

    )
    return gens_w_ferc1_w_label


def _check_merge_eia_ferc(gens_w_ferc1):
    gens_w_ferc1 = (
        gens_w_ferc1.assign(
            plant_fuel_v_unit_diff=lambda x:
                x.opex_nonfuel_per_mwh_plant_fuel
                - x.opex_nonfuel_per_mwh_unit,
        )
    )

    big_diff = gens_w_ferc1[
        (abs(gens_w_ferc1.plant_fuel_v_unit_diff) > 10)
        & (gens_w_ferc1.fuel_type_code_pudl == 'coal')
    ]
    if len(big_diff) > 60:
        warnings.warn(
            "hm, there are too many coal plants with big diffs. "
            f" {len(big_diff)} to be exact. Something may have "
            "have gone wrong in the FERC <> EIA Allocation methods"
        )
    return gens_w_ferc1

################################
# GEM Connection
################################


def get_gem():
    """Grab the GEM data from the inputs directory and do light cleaning."""
    # grab the data, d
    gem = pd.read_excel(PATH_GEM)
    gem = pudl.helpers.simplify_columns(gem).dropna(how='all')
    # lets only look at the columns we need.
    gem = (
        gem[[
            'plant_name', 'unit_name', 'other_ids_location', 'other_ids_unit']]
        .add_suffix('_gem')
    )
    # extract the plant IDs and generator IDS from these two unit columns.
    # assign types! force the plant id into an float before a nullable int
    # hopefully pandas will let you jump straight to ints soon...
    # https://github.com/pandas-dev/pandas/issues/25472
    gem.loc[:, 'plant_id_eia'] = gem.other_ids_location_gem.apply(
        lambda x: literal_eval(x)['EIA']).astype('float').astype('Int64')
    gem.loc[:, 'generator_ids_eia'] = gem.other_ids_unit_gem.apply(
        lambda x: literal_eval(x)['EIA'])
    # set a GEM specific unit id so we can make a table with
    # plants, "units" and generator IDs
    gem = (
        gem.reset_index().rename(columns={'index': 'unit_id_gem'})
        .astype({'unit_id_gem': pd.Int64Dtype()})
        .drop(columns=['other_ids_location_gem', 'other_ids_unit_gem'])
    )
    return gem


def convert_gem_cols_to_gen_ids(gem):
    """Convert the list of generators in a string in a column to a column."""
    split_ids = gem.generator_ids_eia.str.split(
        pat=', ', expand=True).add_prefix('id_')
    units_gem = (
        pd.concat([gem, split_ids], axis=1)
        .melt(
            id_vars=['plant_id_eia', 'plant_name_gem',
                     'unit_name_gem', 'unit_id_gem', ],
            value_vars=split_ids.columns,
            value_name='generator_id')
        .dropna(subset=['generator_id'])
        .drop(columns=['variable'])
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
    )
    return units_gem


def merge_gem_w_df(df, df_source='eia'):
    """Merge GEM unit IDs onto a df w/ plant_id_eia & generator_id cols."""
    gem = get_gem()
    units_gem = convert_gem_cols_to_gen_ids(gem)
    # merge in the GEM IDs into the generator table
    gen_w_gem = (
        pd.merge(
            df,
            units_gem,
            on=['plant_id_eia', 'generator_id'],
            how='outer',
            indicator=True,
            validate='m:1'
        )
    )
    # check to ensure there aren't too many plants in GEM that we
    # are not matching with EIA generators
    missing_plants = gen_w_gem[gen_w_gem._merge == 'right_only']
    if (missing_plants.plant_id_eia.nunique() > 3) & (df_source == 'eia'):
        raise AssertionError(
            "We have more GEM plants that we didn't find EIA matches for than"
            f" expected. {missing_plants.plant_id_eia.nunique()} instead of 3")
    gen_w_gem = gen_w_gem[gen_w_gem._merge != 'right_only']
    gen_w_gem = gen_w_gem.drop(columns=['_merge'])
    return gen_w_gem


################################
# NEMS Connection
################################


def get_nems_headers():
    """
    Get non-duplicate headers for NEMS.

    Returns:
        iterable
    """
    headers = list(
        pd.read_excel(pathlib.Path.cwd().parent / 'inputs' /
                      'pltf860defs_aeo2018.xlsx').T.loc['Definition'])
    counts = Counter(headers)
    suffix_counter = defaultdict(lambda: itertools.count(1))
    headers_non_dupes = [
        elem if counts[elem] == 1
        else elem + f'_{next(suffix_counter[elem])}'
        for elem in headers
    ]
    return headers_non_dupes


def get_nems_year(year):
    """
    Get NEMS data for a single year.

    Args:
        year (integer): four-digit year. Must be a key in NEMS_FILE_NAMES.

    Returns:
        pandas.DataFrame:
    """
    if year not in NEMS_FILE_NAMES:
        raise AssertionError(
            f"{year} not in NEMS_FILE_NAMES map. Add new year and file to "
            "NEMS_FILE_NAMES or try different year. Current years are: "
            f"{NEMS_FILE_NAMES.keys}"
        )
    nems_df = (
        pd.read_csv(
            PATH_INPUTS / NEMS_FILE_NAMES[year],
            delimiter=":",
            names=get_nems_headers(),
            dtype={'Plant ID': pd.Int64Dtype(),
                   'EFD Fuel Codes.1': 'string',  # this is for memory
                   'EFD Fuel Codes.2': 'string',  # mixed string/int cols
                   })
        .assign(report_year=year, report_date=f'{year}-01-01')
    )

    return nems_df


def get_nems():
    """Grab NEMS and perform basic column cleaning."""
    # first grab each year of NEMS data
    nems_dfs = []
    for year in NEMS_FILE_NAMES.keys():
        nems_dfs.append(get_nems_year(year))
    # then squish them together and apply common cleaning
    nems_df = (
        pd.concat(nems_dfs)
        .rename(columns={
            'Plant ID': 'plant_id_eia',
            'Unit ID': 'generator_id',
            'Name Plate Capacity (shared if multiple owners) (MW)':
                'capacity_mw',
            'Average Capacity Factor': 'capacity_factor',
            'Annual Investment in Capital Additions (87$/kW)':
                'capex_annual_per_kw_87',
            'Variable O&M Cost (87$/MWH)': 'variable_om_mwh_87',
            'Fixed O&M Cost (87$/kW)': 'fixed_om_kw_87',
        })
        .astype({
            'report_date': 'datetime64[ns]',
            'report_year': pd.Int64Dtype(),
            'fixed_om_kw_87': 'float32',
            'variable_om_mwh_87': 'float32',
        })
    )
    # theres a ton of trailing zeros on the generator_id column which gotta go
    # this is the only string column we need rn, but beware if others are used
    nems_df.loc[nems_df['generator_id'].notnull(), 'generator_id'] = (
        nems_df.loc[nems_df['generator_id'].notnull(), 'generator_id']
        .astype(str).str.strip()
    )
    return nems_df


def prep_nems():
    """Grab NEMS and groupby plant-fuel.

    Note: There are ~1500 records which have a 0% capacity factor, thus 0 net
    generation, and thus have a calculated fixed cost per MWh of inf.
    """
    # get nems and grouby the gen ids, so we can merge w/ EIA gens
    nems_df = (
        get_nems().groupby(by=IDX_GEN + ['report_year'], as_index=False)
        [['capacity_factor', 'capacity_mw', 'fixed_om_kw_87',
          'variable_om_mwh_87', 'capex_annual_per_kw_87']]
        .mean()
    )
    # Calculate required fields and adjust cost for inflation.
    nems_df = calc_inflation_nems(
        nems_df,
        cols_to_convert=['variable_om_mwh_87',
                         'fixed_om_kw_87', 'capex_annual_per_kw_87'],
        drop=False
    )

    nems_df = (
        nems_df.assign(
            net_generation_mwh_nems=lambda x:
                x.capacity_factor * 8760 * x.capacity_mw,
            fixed_om=lambda x:
                x.fixed_om_kw * 1000 * x.capacity_mw,
            fixed_om_mwh=lambda x: x.fixed_om / x.net_generation_mwh_nems,
            variable_om=lambda x:
                x.variable_om_mwh * x.net_generation_mwh_nems,
            # variable_om_mwh=lambda x: (x.variable_om_mwh),
            fix_var_om_mwh=lambda x: x.variable_om_mwh + x.fixed_om_kw,
            fixed_v_total_ratio=lambda x:
                x.fixed_om / (x.fixed_om + x.variable_om),
            var_v_total_ratio=lambda x:
                x.variable_om / (x.fixed_om + x.variable_om),
            fix_var_om=lambda x: x.fixed_om + x.variable_om
        )
    )

    return nems_df


def _agg_nems_to_plant_fuel(nems_df):
    """
    Aggregate NEMS to `IDX_PLANT_FUEL`.

    This is from EI work- not sure if we need it in this case.
    """
    # aggregate to plant-fuel
    nems_agg = (
        nems_df.groupby(by=IDX_PLANT_FUEL)
        .agg({'variable_om': 'sum',
              'fixed_om': 'sum',
              'capacity_mw': 'sum',
              'net_generation_mwh_nems': 'sum',
              }))
    nems_wtav = weighted_average(
        nems_df,
        {'fixed_om_mwh': 'net_generation_mwh_nems',
         'variable_om_mwh': 'net_generation_mwh_nems',
         'fix_var_om_mwh': 'net_generation_mwh_nems',
         'capex_annual_per_kw': 'capacity_mw',
         # 'fixed_v_total_ratio': 'capacity_mw'
         },
        IDX_PLANT_FUEL)

    nems_cost_df = (
        pd.merge(nems_agg, nems_wtav,
                 on=IDX_PLANT_FUEL,
                 how='outer')
        .assign(
            fixed_v_total_ratio=lambda x:
            x.fixed_om / (x.fixed_om + x.variable_om),
            var_v_total_ratio=lambda x:
            x.variable_om / (x.fixed_om + x.variable_om),
            fix_var_om=lambda x: x.fixed_om + x.variable_om
        )
    )
    return nems_cost_df


def calc_inflation_nems(nems_df_raw, cols_to_convert, drop=True):
    """
    Convert NEMS 1987$ into nominal $s.

    This function calculates inflation using NEMS model to calculate
    nominal fixed and variable costs of NEMS data (reported in 87$ -
    equivalent to 1).

    Args:
        df (pandas.DataFrame): The DataFrame containing the column on which
            you'd like to run an inflation calculation.
        cols_to_convert (list): The names of the column of values you'd like to
            calculate inflation for.
        drop (boolean): default True. If True, drop cols_to_convert before
            returning.
    Returns:
        pd.DataFrame: The new, inflation adjusted values for a given year under
            the same name as the original column.
    """
    # For use with NEMS fixed and variable cost data. Reported in '87' dollars
    # go get the NEMS inflation rates
    nems_idx = pd.read_excel(
        path_nems_infl, header=3, names=['report_year', 'inflation_rate_v_87'])

    # squish them onto raw NEMS df
    nems_df_raw = pd.merge(
        nems_df_raw,
        nems_idx,
        on=['report_year'],
        how='left',
        validate='m:1'
    )

    # create new columns with nominal $s without the _87
    nems_df_raw.loc[:, [c.replace('_87', '') for c in cols_to_convert]] = (
        nems_df_raw.loc[:, cols_to_convert]
        .multiply(nems_df_raw.loc[:, 'inflation_rate_v_87'], axis="index")
        .to_numpy()  # convert to arrary so loc can handle multiple columns
    )
    if drop:
        nems_df_raw = nems_df_raw.drop(
            columns=cols_to_convert + ['inflation_rate_v_87'])
    return nems_df_raw


def add_nems(gens_w_ferc1, pudl_out):
    """Incorporate NEMS aeo2020 data to account for missing FERC O&M costs.

    Args:
        eia_ferc1_merge_df (pandas.DataFrame): A DataFrame containing mcoe
            factors from FERC Form 1 and EIA.
    Returns:
        pandas.DataFrame: A DataFrame with NEMS values added to account for
            missing FERC Form 1 O&M costs.
    """
    nems_merge_df = (
        pd.merge(
            gens_w_ferc1,
            prep_nems().pipe(pudl.helpers.convert_cols_dtypes, 'ferc1'),
            how='left',
            on=IDX_GEN,
            suffixes=("", "_nems"),
            validate='m:1'
        )
        .rename(columns={'capex_annual_per_kw': 'capex_annual_per_kw_nems'})
    )
    missing = (
        len(nems_merge_df[nems_merge_df.plant_id_pudl.isnull()])
        / len(nems_merge_df))
    logger.info(f"NEMS gens w/o mathcing EIA gens: {missing:.1%}")
    if missing > .05:
        raise AssertionError(
            "Too many NEMS generators that don't match with EIA gens. Check "
            "`prep_nems()`"
        )
    return nems_merge_df


def get_average_fix_v_var_ratios(gens_w_ferc1_nems):
    """Calc average fix/var ratios."""
    # we want average fix/var ratios by fuel type and year
    # we also want the standard deviations so we can know
    # what the dispersion of values is like
    ft_gb = (
        gens_w_ferc1_nems
        .groupby(['fuel_type_code_pudl', 'report_date'])
        [['fixed_v_total_ratio', 'var_v_total_ratio']]
    )
    ft_fix_var_avg = (
        pd.merge(
            ft_gb.mean().dropna(),
            ft_gb.std().dropna(),
            right_index=True, left_index=True,
            suffixes=('_avg', '_std')
        )
    )
    # we're going to remove the fuel types with high std's
    # so we can feel comfy about assuming the mean
    ft_fix_var_avg = (
        ft_fix_var_avg
        .loc[ft_fix_var_avg.fixed_v_total_ratio_std < .1]
        .reset_index()
        .drop(columns=['fixed_v_total_ratio_std', 'var_v_total_ratio_std'])
    )
    return ft_fix_var_avg


def fill_in_opex_w_nems(gens_w_ferc1_nems):
    """
    Fill in the opex and split into fixed/variable with NEMS.

    Args:
        gens_w_ferc1_nems (pandas.DataFrame): result of `add_nems()`
    """
    ft_fix_var_avg = get_average_fix_v_var_ratios(gens_w_ferc1_nems)

    gens_w_ferc1_nems_filled = (
        gens_w_ferc1_nems
        .merge(
            ft_fix_var_avg,
            on=['fuel_type_code_pudl', 'report_date'],
            how='left',
            validate='m:1'
        )
        .assign(
            opex_nonfuel_per_mwh=lambda x:
                x.opex_nonfuel_per_mwh.fillna(
                    x.variable_om_mwh + x.fixed_om_mwh),
            opex_fixed_per_mwh=lambda x: np.where(
                x.fixed_v_total_ratio.notnull(),
                x.opex_nonfuel_per_mwh * x.fixed_v_total_ratio,
                x.opex_nonfuel_per_mwh * x.fixed_v_total_ratio_avg,),
            opex_variable_per_mwh=lambda x: np.where(
                x.var_v_total_ratio.notnull(),
                x.opex_nonfuel_per_mwh * x.var_v_total_ratio,
                x.opex_nonfuel_per_mwh * x.var_v_total_ratio_avg,),
        )
    )
    return gens_w_ferc1_nems_filled

################################
# CEMS
################################


def get_cems(epacems_path, years=()):
    """Get annual CEMS data."""
    # A list of the columns you'd like to include in your analysis
    idx_cols_cems = [
        'year', 'plant_id_eia', 'unitid'
    ]
    years_filter = pudl.output.epacems.year_state_filter(years=years)

    # Select emissions data are grouped by state, plant_id and unit_id
    # Remember to change the datatype for 'state' from category to string
    my_cems_dd = (
        dd.read_parquet(
            epacems_path,
            columns=idx_cols_cems + COLS_SUM_CEMS,
            filters=years_filter,
        )
        .astype({'year': int})
        .groupby(idx_cols_cems)
        [COLS_SUM_CEMS]
        .sum()
    ).reset_index()
    cems_by_boiler = (
        my_cems_dd.compute()
        .rename(
            columns={
                'unitid': 'boiler_id',
                'year': 'report_year'}
        )
        .assign(report_date=lambda x:
                pd.to_datetime(x.report_year, format='%Y'))
    )
    return cems_by_boiler


def stuff(cems_by_boiler, gen, pudl_out, unit_id_col):
    """Do cems stuff."""
    idx_unit_cols = ['plant_id_eia', 'report_date', unit_id_col]

    eia_with_boiler_id = (
        pudl_out.bga()[IDX_BOILER + ['unit_id_pudl', 'generator_id']]
        .pipe(merge_gem_w_df, df_source='bga')
        [IDX_BOILER + [unit_id_col]]
        .drop_duplicates()
    )

    # Add boiler id to EIA data. Boilder id matches (almost) with CEMS unitid.
    eia_cems_merge = (
        pd.merge(
            eia_with_boiler_id,
            cems_by_boiler,
            on=IDX_BOILER,
            how='right',
            validate='m:1'
        )
        .groupby(idx_unit_cols, dropna=False)[COLS_SUM_CEMS]
        .sum(min_count=1)
        .reset_index()
        .assign(so2_mass_tons=lambda x: x.so2_mass_lbs / 2000,
                nox_mass_tons=lambda x: x.nox_mass_lbs / 2000)
        .drop(['so2_mass_lbs', 'nox_mass_lbs'], axis=1)
        .pipe(pudl.helpers.convert_cols_dtypes, 'eia')
        .merge(
            gen.drop_duplicates(subset=idx_unit_cols)
            [idx_unit_cols + ['operational_status', 'sector_name',
                              'fuel_type_code_pudl']],
            on=idx_unit_cols,
            validate='1:1',
            how='left'
        )
        .merge(
            gen.groupby(idx_unit_cols, dropna=False, as_index=False)
            .agg({'unit_id_eia': str_squish, 'generator_id': str_squish}),
            on=idx_unit_cols,
            how='left',
            validate='1:1'
        )
    )
    return eia_cems_merge


################################
# General helper functions
################################


def weighted_average(df, wa_col_dict, idx_cols):
    """Generate a weighted average for multiple columns at once.

    When aggregating the data by plant and fuel type, many of the values can
    be summed. Heat rates and generator age, however, are claculated with a
    weighted average. This function exists because there is no python or numpy
    function to calculate weighted average like there is for .sum() or .mean().

    In this case, the heat rate calculation is based the 'weight' or share
    of generator net generation (net_generation_mwh) and the generator age is
    based on that of the generators' capacity (capacity_mw). As seen in the
    global eia_wa_col_dict dictionary.

    Args:
        df (pandas.DataFrame): A DataFrame containing, at minimum, the columns
            specified in the other parameters wa_col_dict and by_cols.
        wa_col_dict (dict): A dictionary containing keys and values that
            represent the column names for the 'data' and 'weight' values.
        idx_cols (list): A list of the columns to group by when calcuating
            the weighted average value.
    Returns:
        pandas.DataFrame: A DataFrame containing weigted average values for
            specified 'data' columns based on specified 'weight' columns.
            Grouped by an indicated set of columns.
    """
    merge_df = df.loc[:, idx_cols]
    g = df.groupby(idx_cols, dropna=False)
    for data, weight in wa_col_dict.items():
        logger.debug(' - Calculating weighted average for ' + data)
        df.loc[:, '_data_times_weight'] = df.loc[:, data] * df.loc[:, weight]
        df.loc[:, '_weight_where_notnull'] = (
            df.loc[:, weight] * pd.notnull(df.loc[:, data]))
        result = g['_data_times_weight'].sum(
        ) / g['_weight_where_notnull'].sum()
        del df['_data_times_weight'], df['_weight_where_notnull']
        result = result.copy().to_frame(name=data).reset_index()
        merge_df = pd.merge(merge_df.copy(), result, on=idx_cols, how='outer')
    return merge_df.drop_duplicates()


composite_id_assign = composite_unit_gen_id = (
    lambda z: np.where(
        z.unit_id_pudl.notnull() | z.generator_id.notnull(),
        (z.unit_id_pudl.astype(pd.StringDtype())
         .fillna("genid-" + z.generator_id.astype(str))),
        pd.NA
    )
)


def str_squish(x):
    """Squish strings from a groupby into a list."""
    return '; '.join(list(map(str, [x for x in x.unique() if x is not pd.NA])))
