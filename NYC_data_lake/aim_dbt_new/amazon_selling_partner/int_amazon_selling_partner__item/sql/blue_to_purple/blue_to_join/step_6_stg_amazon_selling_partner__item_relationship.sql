

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_relationship_base"
),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    child_asin
    
 as 
    
    child_asin
    
, 
    
    
    parent_asin
    
 as 
    
    parent_asin
    
, 
    
    
    type
    
 as 
    
    type
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(child_asin as TEXT) as child_asin,
        cast(parent_asin as TEXT) as parent_asin,
        upper(type) as type
    from fields
)

select *
from final