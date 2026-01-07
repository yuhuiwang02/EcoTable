

with base as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__campaign_report_hourly_tmp"
), 

fields as (

    select
        
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    average_video_play
    
 as 
    
    average_video_play
    
, 
    
    
    average_video_play_per_user
    
 as 
    
    average_video_play_per_user
    
, 
    
    
    clicks
    
 as 
    
    clicks
    
, 
    
    
    comments
    
 as 
    
    comments
    
, 
    
    
    conversion
    
 as 
    
    conversion
    
, 
    
    
    conversion_rate
    
 as 
    
    conversion_rate
    
, 
    
    
    cost_per_conversion
    
 as 
    
    cost_per_conversion
    
, 
    
    
    cpc
    
 as 
    
    cpc
    
, 
    
    
    cpm
    
 as 
    
    cpm
    
, 
    
    
    ctr
    
 as 
    
    ctr
    
, 
    
    
    follows
    
 as 
    
    follows
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    likes
    
 as 
    
    likes
    
, 
    
    
    profile_visits
    
 as 
    
    profile_visits
    
, 
    
    
    reach
    
 as 
    
    reach
    
, 
    
    
    real_time_conversion
    
 as 
    
    real_time_conversion
    
, 
    
    
    shares
    
 as 
    
    shares
    
, 
    
    
    spend
    
 as 
    
    spend
    
, 
    
    
    stat_time_hour
    
 as 
    
    stat_time_hour
    
, 
    
    
    total_purchase_value
    
 as 
    
    total_purchase_value
    
, 
    
    
    total_sales_lead_value
    
 as 
    
    total_sales_lead_value
    
, 
    
    
    video_play_actions
    
 as 
    
    video_play_actions
    
, 
    
    
    video_views_p_25
    
 as 
    
    video_views_p_25
    
, 
    
    
    video_views_p_50
    
 as 
    
    video_views_p_50
    
, 
    
    
    video_views_p_75
    
 as 
    
    video_views_p_75
    
, 
    
    
    video_watched_2_s
    
 as 
    
    video_watched_2_s
    
, 
    
    
    video_watched_6_s
    
 as 
    
    video_watched_6_s
    



    
        


, cast('' as TEXT) as source_relation




    from base
), 

final as (

    select
        source_relation,  
        campaign_id,
        cast(stat_time_hour as timestamp) as stat_time_hour,
        cpc, 
        cpm,
        ctr,
        coalesce(impressions, 0) as impressions,
        coalesce(clicks, 0) as clicks, 
        coalesce(spend, 0) as spend, 
        reach,
        coalesce(conversion, 0) as conversion,
        cost_per_conversion,
        conversion_rate,
        coalesce(likes, 0) as likes,
        coalesce(comments, 0) as comments,
        coalesce(shares, 0) as shares,
        coalesce(profile_visits, 0) as profile_visits,
        coalesce(follows, 0) as follows,
        coalesce(video_play_actions, 0) as video_play_actions,
        coalesce(video_watched_2_s, 0) as video_watched_2_s,
        coalesce(video_watched_6_s, 0) as video_watched_6_s,
        coalesce(video_views_p_25, 0) as video_views_p_25,
        coalesce(video_views_p_50, 0) as video_views_p_50,
        coalesce(video_views_p_75, 0) as video_views_p_75,
        average_video_play,
        average_video_play_per_user,
        coalesce(real_time_conversion, 0) as real_time_conversion,
        coalesce(total_purchase_value, 0) as total_purchase_value,
        coalesce(total_sales_lead_value, 0) as total_sales_lead_value

        





    from fields
)

select *
from final