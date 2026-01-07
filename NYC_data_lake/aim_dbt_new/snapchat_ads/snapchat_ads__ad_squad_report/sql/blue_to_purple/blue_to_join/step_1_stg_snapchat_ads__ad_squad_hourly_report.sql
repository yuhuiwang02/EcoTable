

with base as (

    select * 
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_squad_hourly_report_tmp"
),

fields as (

    select
        
    
    
    ad_squad_id
    
 as 
    
    ad_squad_id
    
, 
    
    
    attachment_quartile_1
    
 as 
    
    attachment_quartile_1
    
, 
    
    
    attachment_quartile_2
    
 as 
    
    attachment_quartile_2
    
, 
    
    
    attachment_quartile_3
    
 as 
    
    attachment_quartile_3
    
, 
    
    
    attachment_total_view_time_millis
    
 as 
    
    attachment_total_view_time_millis
    
, 
    
    
    attachment_view_completion
    
 as 
    
    attachment_view_completion
    
, 
    
    
    date
    
 as 
    
    date
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    quartile_1
    
 as 
    
    quartile_1
    
, 
    
    
    quartile_2
    
 as 
    
    quartile_2
    
, 
    
    
    quartile_3
    
 as 
    
    quartile_3
    
, 
    
    
    saves
    
 as 
    
    saves
    
, 
    
    
    screen_time_millis
    
 as 
    
    screen_time_millis
    
, 
    
    
    shares
    
 as 
    
    shares
    
, 
    
    
    spend
    
 as 
    
    spend
    
, 
    
    
    swipes
    
 as 
    
    swipes
    
, 
    
    
    video_views
    
 as 
    
    video_views
    
, 
    
    
    view_completion
    
 as 
    
    view_completion
    
, 
    
    
    view_time_millis
    
 as 
    
    view_time_millis
    
, 
    
    
    conversion_purchases_value
    
 as 
    
    conversion_purchases_value
    
, 
    
    
    conversion_purchases
    
 as 
    
    conversion_purchases
    


        
    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        ad_squad_id,
        cast (date as timestamp) as date_hour,
        attachment_quartile_1,
        attachment_quartile_2,
        attachment_quartile_3,
        (attachment_total_view_time_millis / 1000000.0) as attachment_total_view_time,
        attachment_view_completion,
        quartile_1,
        quartile_2,
        quartile_3,
        saves,
        shares,
        (screen_time_millis / 1000000.0) as screen_time,
        video_views,
        view_completion,
        (view_time_millis / 1000000.0) as view_time,
        impressions,
        (spend / 1000000.0) as spend,
        swipes,
        coalesce(cast(conversion_purchases_value as float), 0) / 1000000.0 as conversion_purchases_value

        
            , coalesce(cast(conversion_purchases as bigint), 0) as conversion_purchases
        

        





    from fields
)

select *
from final