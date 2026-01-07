{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__item_relationship_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__item_relationship_base')),
                staging_columns=get_item_relationship_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(child_asin as {{ dbt.type_string() }}) as child_asin,
        cast(parent_asin as {{ dbt.type_string() }}) as parent_asin,
        upper(type) as type
    from fields
)

select *
from final
