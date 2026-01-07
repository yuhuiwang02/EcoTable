

with base as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__ad_group_history_tmp"
), 

fields as (

    select
        
    
    
    action_days
    
 as 
    
    action_days
    
, 
    
    
    adgroup_id
    
 as 
    
    adgroup_id
    
, 
    
    
    adgroup_name
    
 as 
    
    adgroup_name
    
, 
    
    
    advertiser_id
    
 as 
    
    advertiser_id
    
, 
    
    
    audience_type
    
 as 
    
    audience_type
    
, 
    
    
    budget
    
 as 
    
    budget
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    category
    
 as 
    
    category
    
, 
    
    
    display_name
    
 as 
    
    display_name
    
, 
    
    
    frequency
    
 as 
    
    frequency
    
, 
    
    
    frequency_schedule
    
 as 
    
    frequency_schedule
    
, 
    
    
    gender
    
 as 
    
    gender
    
, 
    
    
    landing_page_url
    
 as 
    
    landing_page_url
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    
, 
    
    
    interest_category_v_2
    
 as 
    
    interest_category_v_2
    
, 
    
    
    action_categories
    
 as 
    
    action_categories
    
, 
    
    
    age_groups
    
 as 
    
    age_groups
    
, 
    
    
    languages
    
 as 
    
    languages
    




    
        


, cast('' as TEXT) as source_relation




    from base
), 

final as (

    select
        source_relation,
        adgroup_id as ad_group_id,
        cast(updated_at as timestamp) as updated_at,
        advertiser_id,
        campaign_id,
        action_days,
        action_categories,
        adgroup_name as ad_group_name,
        age_groups,
        audience_type,
        budget,
        category,
        display_name,
        interest_category_v_2 as interest_category,
        frequency,
        frequency_schedule,
        gender,
        languages, 
        landing_page_url,
        row_number() over (partition by source_relation, adgroup_id order by updated_at desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final