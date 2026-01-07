{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__item_image_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__item_image_base')),
                staging_columns=get_item_image_columns()
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
        height,
        link,
        marketplace_id,
        upper(variant) as variant,
        width
    from fields
)

select *
from final
