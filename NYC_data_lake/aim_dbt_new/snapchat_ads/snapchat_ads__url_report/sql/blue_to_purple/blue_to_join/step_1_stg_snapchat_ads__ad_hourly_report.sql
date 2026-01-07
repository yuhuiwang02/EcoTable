

with base as (

    select * 
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_hourly_report_tmp"
),

fields as (

    select
        
    
    
    ad_id
    
 as 
    
    ad_id
    
, 
    cast(null as numeric(28,6)) as 
    
    attachment_quartile_1
    
 , 
    cast(null as numeric(28,6)) as 
    
    attachment_quartile_2
    
 , 
    cast(null as numeric(28,6)) as 
    
    attachment_quartile_3
    
 , 
    cast(null as numeric(28,6)) as 
    
    attachment_total_view_time_millis
    
 , 
    cast(null as numeric(28,6)) as 
    
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
    cast(null as numeric(28,6)) as 
    
    quartile_1
    
 , 
    cast(null as numeric(28,6)) as 
    
    quartile_2
    
 , 
    cast(null as numeric(28,6)) as 
    
    quartile_3
    
 , 
    cast(null as numeric(28,6)) as 
    
    saves
    
 , 
    cast(null as numeric(28,6)) as 
    
    screen_time_millis
    
 , 
    cast(null as numeric(28,6)) as 
    
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
    cast(null as numeric(28,6)) as 
    
    video_views
    
 , 
    cast(null as numeric(28,6)) as 
    
    view_completion
    
 , 
    cast(null as numeric(28,6)) as 
    
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
        ad_id,
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