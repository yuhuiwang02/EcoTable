{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__item_summary_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__item_summary_base')),
                staging_columns=get_item_summary_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        adult_product as is_adult_product,
        cast(asin as {{ dbt.type_string() }}) as asin,
        autographed as is_autographed,
        brand,
        classification_id,
        color,
        contributors,
        display_name,
        item_classification,
        item_name,
        manufacturer,
        marketplace_id,
        memorabilia as is_memorabilia,
        model_number,
        package_quantity,
        part_number,
        release_date,
        size,
        style,
        trade_in_eligible as is_trade_in_eligible,
        website_display_group,
        website_display_group_name
    from fields
)

select *
from final
