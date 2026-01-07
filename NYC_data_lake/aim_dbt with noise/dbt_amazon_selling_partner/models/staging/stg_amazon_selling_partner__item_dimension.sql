{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__item_dimension_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__item_dimension_base')),
                staging_columns=get_item_dimension_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as {{ dbt.type_string() }}) as asin,
        marketplace_id,
        item_height_unit,
        item_height_value,
        item_length_unit,
        item_length_value,
        item_weight_unit,
        item_weight_value,
        item_width_unit,
        item_width_value,
        package_height_unit,
        package_height_value,
        package_length_unit,
        package_length_value,
        package_weight_unit,
        package_weight_value,
        package_width_unit,
        package_width_value
    from fields
)

select *
from final
