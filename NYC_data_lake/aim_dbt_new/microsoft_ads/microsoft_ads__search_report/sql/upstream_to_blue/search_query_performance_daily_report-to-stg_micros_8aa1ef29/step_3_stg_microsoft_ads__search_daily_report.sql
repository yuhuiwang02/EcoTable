

with base as (

    select * 
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__search_daily_report_tmp"
),

fields as (

    select
        
    
    
    account_id
    
 as 
    
    account_id
    
, 
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    ad_id
    
 as 
    
    ad_id
    
, 
    
    
    bid_match_type
    
 as 
    
    bid_match_type
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    clicks
    
 as 
    
    clicks
    
, 
    
    
    date
    
 as 
    
    date
    
, 
    
    
    delivered_match_type
    
 as 
    
    delivered_match_type
    
, 
    
    
    device_os
    
 as 
    
    device_os
    
, 
    
    
    device_type
    
 as 
    
    device_type
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    keyword_id
    
 as 
    
    keyword_id
    
, 
    
    
    language
    
 as 
    
    language
    
, 
    
    
    network
    
 as 
    
    network
    
, 
    
    
    search_query
    
 as 
    
    search_query
    
, 
    
    
    spend
    
 as 
    
    spend
    
, 
    
    
    top_vs_other
    
 as 
    
    top_vs_other
    
, 
    
    
    conversions_qualified
    
 as 
    
    conversions_qualified
    
, 
    
    
    conversions
    
 as 
    
    conversions
    
, 
    
    
    revenue
    
 as 
    
    revenue
    
, 
    
    
    all_conversions
    
 as 
    
    all_conversions
    
, 
    
    
    all_conversions_qualified
    
 as 
    
    all_conversions_qualified
    
, 
    
    
    all_revenue
    
 as 
    
    all_revenue
    




    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        date as date_day,
        account_id,
        campaign_id,
        ad_group_id,
        ad_id,
        keyword_id,
        search_query,
        device_os,
        device_type,
        network,
        language,
        bid_match_type,
        delivered_match_type,
        top_vs_other,
        coalesce(clicks, 0) as clicks, 
        coalesce(impressions, 0) as impressions,
        coalesce(spend, 0) as spend,
        coalesce(coalesce(cast(conversions_qualified as integer), cast(conversions as integer)), 0) as conversions,
        coalesce(cast(revenue as float), 0) as conversions_value,
        coalesce(coalesce(cast(all_conversions_qualified as integer), cast(all_conversions as integer)), 0) as all_conversions,
        -- sometimes this field comes in as a string
        coalesce(cast(replace(cast(all_revenue as TEXT), ',', '') as float), 0) as all_conversions_value

        





    from fields
)

select * 
from final