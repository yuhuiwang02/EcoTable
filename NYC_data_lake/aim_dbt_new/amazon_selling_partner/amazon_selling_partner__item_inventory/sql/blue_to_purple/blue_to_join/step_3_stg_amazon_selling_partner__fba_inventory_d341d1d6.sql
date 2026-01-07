

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__fba_inventory_researching_base"
),

fields as (

    select
        
    
    
    inventory_summary_id
    
 as 
    
    inventory_summary_id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    quantity
    
 as 
    
    quantity
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
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