

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__product_ad_history_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    asin
    
 as 
    
    asin
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    creation_date
    
 as 
    
    creation_date
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    last_updated_date
    
 as 
    
    last_updated_date
    
, 
    
    
    serving_status
    
 as 
    
    serving_status
    
, 
    
    
    sku
    
 as 
    
    sku
    
, 
    
    
    state
    
 as 
    
    state
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(id as TEXT) as ad_id,
        cast(ad_group_id as TEXT) as ad_group_id,
        asin,
        cast(campaign_id as TEXT) as campaign_id,
        creation_date,
        last_updated_date,
        serving_status,
        sku,
        state,
        row_number() over (partition by source_relation, id order by last_updated_date desc) = 1 as is_most_recent_record
    from fields
)

select *
from final