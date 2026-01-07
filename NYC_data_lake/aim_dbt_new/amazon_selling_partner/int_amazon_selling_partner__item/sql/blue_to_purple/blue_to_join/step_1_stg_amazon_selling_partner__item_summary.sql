

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_summary_base"
),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    adult_product
    
 as 
    
    adult_product
    
, 
    
    
    asin
    
 as 
    
    asin
    
, 
    
    
    autographed
    
 as 
    
    autographed
    
, 
    
    
    brand
    
 as 
    
    brand
    
, 
    
    
    classification_id
    
 as 
    
    classification_id
    
, 
    
    
    color
    
 as 
    
    color
    
, 
    
    
    contributors
    
 as 
    
    contributors
    
, 
    
    
    display_name
    
 as 
    
    display_name
    
, 
    
    
    item_classification
    
 as 
    
    item_classification
    
, 
    
    
    item_name
    
 as 
    
    item_name
    
, 
    
    
    manufacturer
    
 as 
    
    manufacturer
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    
, 
    
    
    memorabilia
    
 as 
    
    memorabilia
    
, 
    
    
    model_number
    
 as 
    
    model_number
    
, 
    
    
    package_quantity
    
 as 
    
    package_quantity
    
, 
    
    
    part_number
    
 as 
    
    part_number
    
, 
    
    
    release_date
    
 as 
    
    release_date
    
, 
    
    
    size
    
 as 
    
    size
    
, 
    
    
    style
    
 as 
    
    style
    
, 
    
    
    trade_in_eligible
    
 as 
    
    trade_in_eligible
    
, 
    
    
    website_display_group
    
 as 
    
    website_display_group
    
, 
    
    
    website_display_group_name
    
 as 
    
    website_display_group_name
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        adult_product as is_adult_product,
        cast(asin as TEXT) as asin,
        autographed as is_autographed,
        brand,
        classification_id,
        color,
        contributors,
        display_name,
        item_classification,
        item_name,
        manufacturer,
        marketplace_id,
        memorabilia as is_memorabilia,
        model_number,
        package_quantity,
        part_number,
        release_date,
        size,
        style,
        trade_in_eligible as is_trade_in_eligible,
        website_display_group,
        website_display_group_name
    from fields
)

select *
from final