

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_display_group_sales_rank_base"
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
    
, 
    
    
    website_display_group
    
 as 
    
    website_display_group
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        link,
        rank,
        title,
        website_display_group
    from fields
)

select *
from final