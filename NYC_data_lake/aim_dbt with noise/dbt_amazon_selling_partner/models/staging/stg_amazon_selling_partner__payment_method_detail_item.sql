{{ config(enabled=var('amazon_selling_partner__using_orders_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__payment_method_detail_item_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__payment_method_detail_item_base')),
                staging_columns=get_payment_method_detail_item_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        amazon_order_id,
        method
    from fields
)

select *
from final
