

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_image_base"
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
    
    
    height
    
 as 
    
    height
    
, 
    
    
    link
    
 as 
    
    link
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    
, 
    
    
    variant
    
 as 
    
    variant
    
, 
    
    
    width
    
 as 
    
    width
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        height,
        link,
        marketplace_id,
        upper(variant) as variant,
        width
    from fields
)

select *
from final