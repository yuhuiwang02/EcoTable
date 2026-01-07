

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_classification_sales_rank_base"
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
    
    
    classification_id
    
 as 
    
    classification_id
    
, 
    
    
    link
    
 as 
    
    link
    
, 
    
    
    rank
    
 as 
    
    rank
    
, 
    
    
    title
    
 as 
    
    title
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        classification_id,
        link,
        rank,
        title
    from fields
)

select *
from final