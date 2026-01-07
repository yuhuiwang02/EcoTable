

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__keyword_history_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    bid
    
 as 
    
    bid
    
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
    
    
    keyword_text
    
 as 
    
    keyword_text
    
, 
    
    
    last_updated_date
    
 as 
    
    last_updated_date
    
, 
    
    
    match_type
    
 as 
    
    match_type
    
, 
    
    
    native_language_keyword
    
 as 
    
    native_language_keyword
    
, 
    
    
    serving_status
    
 as 
    
    serving_status
    
, 
    
    
    state
    
 as 
    
    state
    
, 
    
    
    native_language_locale
    
 as 
    
    native_language_locale
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(id as TEXT) as keyword_id,
        cast(ad_group_id as TEXT) as ad_group_id,
        bid,
        cast(campaign_id as TEXT) as campaign_id,
        creation_date,
        keyword_text,
        last_updated_date,
        match_type,
        native_language_keyword,
        serving_status,
        state,
        native_language_locale,
        row_number() over (partition by source_relation, id order by last_updated_date desc) = 1 as is_most_recent_record
    from fields
)

select *
from final