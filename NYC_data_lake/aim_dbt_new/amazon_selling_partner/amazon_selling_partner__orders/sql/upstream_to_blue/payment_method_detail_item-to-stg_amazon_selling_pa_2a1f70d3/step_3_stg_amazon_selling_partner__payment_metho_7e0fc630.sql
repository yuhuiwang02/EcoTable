

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__payment_method_detail_item_base"
),

fields as (

    select
        
    
    
    amazon_order_id
    
 as 
    
    amazon_order_id
    
, 
    
    
    method
    
 as 
    
    method
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
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