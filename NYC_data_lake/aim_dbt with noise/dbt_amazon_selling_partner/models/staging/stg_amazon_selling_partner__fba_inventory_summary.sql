{{ config(enabled=var('amazon_selling_partner__using_fba_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__fba_inventory_summary_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__fba_inventory_summary_base')),
                staging_columns=get_fba_inventory_summary_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_id as inventory_summary_id,
        cast(asin as {{ dbt.type_string() }}) as asin,
        fn_sku,
        seller_sku,
        product_name,
        condition,
        last_updated_time as last_updated_at,
        total_quantity,
        total_researching_quantity,
        total_reserved_quantity,
        fullfillable_quantity,
        total_unfulfillable_quantity,
        pending_customer_order_quantity,
        pending_transshipment_quantity,
        fc_processing_quantity,
        inblound_shipped_quantity,
        inbound_receiving_quantity,
        inbound_working_quantity,
        warehouse_damaged_quantity,
        carrier_damaged_quantity,
        customer_damaged_quantity,
        defective_quantity,
        distributor_damaged_quantity,
        expired_quantity,
        granularity_id,
        granularity_type

    from fields
)

select *
from final
