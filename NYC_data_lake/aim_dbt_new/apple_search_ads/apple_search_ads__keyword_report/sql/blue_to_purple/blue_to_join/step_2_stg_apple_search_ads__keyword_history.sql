

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__keyword_history_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    bid_amount
    
 as 
    
    bid_amount
    
, 
    
    
    bid_currency
    
 as 
    
    bid_currency
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    match_type
    
 as 
    
    match_type
    
, 
    
    
    modification_time
    
 as 
    
    modification_time
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    text
    
 as 
    
    text
    



        
    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        modification_time as modified_at,
        campaign_id,
        ad_group_id,
        id as keyword_id,
        bid_amount, 
        bid_currency,
        match_type,
        status as keyword_status,
        text as keyword_text,
        row_number() over (partition by source_relation, id order by modification_time desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final