

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__fba_inventory_summary_base"
),

fields as (

    select
        
    
    
    _fivetran_id
    
 as 
    
    _fivetran_id
    
, 
    
    
    asin
    
 as 
    
    asin
    
, 
    
    
    carrier_damaged_quantity
    
 as 
    
    carrier_damaged_quantity
    
, 
    
    
    condition
    
 as 
    
    condition
    
, 
    
    
    customer_damaged_quantity
    
 as 
    
    customer_damaged_quantity
    
, 
    
    
    defective_quantity
    
 as 
    
    defective_quantity
    
, 
    
    
    distributor_damaged_quantity
    
 as 
    
    distributor_damaged_quantity
    
, 
    
    
    expired_quantity
    
 as 
    
    expired_quantity
    
, 
    
    
    fc_processing_quantity
    
 as 
    
    fc_processing_quantity
    
, 
    
    
    fn_sku
    
 as 
    
    fn_sku
    
, 
    
    
    fullfillable_quantity
    
 as 
    
    fullfillable_quantity
    
, 
    
    
    granularity_id
    
 as 
    
    granularity_id
    
, 
    
    
    granularity_type
    
 as 
    
    granularity_type
    
, 
    
    
    inblound_shipped_quantity
    
 as 
    
    inblound_shipped_quantity
    
, 
    
    
    inbound_receiving_quantity
    
 as 
    
    inbound_receiving_quantity
    
, 
    
    
    inbound_working_quantity
    
 as 
    
    inbound_working_quantity
    
, 
    
    
    last_updated_time
    
 as 
    
    last_updated_time
    
, 
    
    
    pending_customer_order_quantity
    
 as 
    
    pending_customer_order_quantity
    
, 
    
    
    pending_transshipment_quantity
    
 as 
    
    pending_transshipment_quantity
    
, 
    
    
    product_name
    
 as 
    
    product_name
    
, 
    
    
    seller_sku
    
 as 
    
    seller_sku
    
, 
    
    
    total_quantity
    
 as 
    
    total_quantity
    
, 
    
    
    total_researching_quantity
    
 as 
    
    total_researching_quantity
    
, 
    
    
    total_reserved_quantity
    
 as 
    
    total_reserved_quantity
    
, 
    
    
    total_unfulfillable_quantity
    
 as 
    
    total_unfulfillable_quantity
    
, 
    
    
    warehouse_damaged_quantity
    
 as 
    
    warehouse_damaged_quantity
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_id as inventory_summary_id,
        cast(asin as TEXT) as asin,
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