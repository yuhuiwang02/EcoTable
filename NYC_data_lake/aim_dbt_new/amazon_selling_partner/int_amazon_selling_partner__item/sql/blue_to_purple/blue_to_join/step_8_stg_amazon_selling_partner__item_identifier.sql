

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_identifier_base"
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
    
    
    identifier
    
 as 
    
    identifier
    
, 
    
    
    identifier_type
    
 as 
    
    identifier_type
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        identifier,
        identifier_type,
        marketplace_id
    from fields
)

select *
from final