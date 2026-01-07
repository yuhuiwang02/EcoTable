{{ config(enabled=var('amazon_selling_partner__using_fba_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__fba_inventory_researching_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__fba_inventory_researching_base')),
                staging_columns=get_fba_inventory_researching_quantity_entry_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        inventory_summary_id,
        name,
        quantity
    from fields
)

select *
from final
