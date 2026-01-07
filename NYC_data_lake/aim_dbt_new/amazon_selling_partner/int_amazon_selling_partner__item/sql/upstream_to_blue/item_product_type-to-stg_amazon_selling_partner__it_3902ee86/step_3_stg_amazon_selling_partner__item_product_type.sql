

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_product_type_base"
),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    asin
    
 as 
    
    asin
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    
, 
    
    
    product_type
    
 as 
    
    product_type
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        marketplace_id,
        product_type
    from fields
)

select *
from final