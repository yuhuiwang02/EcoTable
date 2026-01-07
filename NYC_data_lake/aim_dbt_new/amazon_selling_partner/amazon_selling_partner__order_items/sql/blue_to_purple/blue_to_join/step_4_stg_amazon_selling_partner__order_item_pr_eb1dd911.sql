

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__order_item_promotion_id_base"
),

fields as (

    select
        
    
    
    amazon_order_id
    
 as 
    
    amazon_order_id
    
, 
    
    
    order_item_id
    
 as 
    
    order_item_id
    
, 
    
    
    promotion_id
    
 as 
    
    promotion_id
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        amazon_order_id,
        order_item_id,
        promotion_id
    from fields
)

select *
from final