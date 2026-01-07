

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__ad_group_history_tmp"
),

fields as (

    select
        
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    creation_date
    
 as 
    
    creation_date
    
, 
    
    
    default_bid
    
 as 
    
    default_bid
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    last_updated_date
    
 as 
    
    last_updated_date
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    serving_status
    
 as 
    
    serving_status
    
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
        cast(id as TEXT) as ad_group_id,
        cast(campaign_id as TEXT) as campaign_id,
        creation_date,
        default_bid,
        last_updated_date,
        name as ad_group_name,
        serving_status,
        state,
        row_number() over (partition by source_relation, id order by last_updated_date desc) = 1 as is_most_recent_record
    from fields
)

select *
from final