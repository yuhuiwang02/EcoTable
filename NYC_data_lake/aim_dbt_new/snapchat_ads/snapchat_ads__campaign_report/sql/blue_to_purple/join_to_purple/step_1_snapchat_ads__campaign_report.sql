

with campaign_hourly as (

    select *,
        
            conversion_purchases as total_conversions
        
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__campaign_hourly_report"

), account as (

    select *
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_account_history"
    where is_most_recent_record = true

), campaigns as (

    select *
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__campaign_history"
    where is_most_recent_record = true


), aggregated as (

    select
        campaign_hourly.source_relation,
        cast(campaign_hourly.date_hour as date) as date_day,
        account.ad_account_id,
        account.ad_account_name,
        campaign_hourly.campaign_id,
        campaigns.campaign_name,
        account.currency,
        sum(campaign_hourly.swipes) as swipes,
        sum(campaign_hourly.impressions) as impressions,
        round(sum(campaign_hourly.spend),2) as spend,
        sum(campaign_hourly.total_conversions) as total_conversions,
        round(cast(sum(campaign_hourly.conversion_purchases_value) as numeric(28,6)), 2) as conversion_purchases_value

        



    
    



    



    
    
    
        
        , sum(coalesce(conversion_purchases, 0)) as conversion_purchases
        

    



        
        





    





    
    from campaign_hourly
    left join campaigns
        on campaign_hourly.campaign_id = campaigns.campaign_id
        and campaign_hourly.source_relation = campaigns.source_relation
    left join account
        on campaigns.ad_account_id = account.ad_account_id
        and campaigns.source_relation = account.source_relation
    
    group by 1,2,3,4,5,6,7

)

select *
from aggregated