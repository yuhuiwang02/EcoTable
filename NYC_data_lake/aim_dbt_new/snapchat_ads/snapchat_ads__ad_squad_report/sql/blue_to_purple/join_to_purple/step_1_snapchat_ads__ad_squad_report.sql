

with ad_squad_hourly as (

    select *,
        
            conversion_purchases as total_conversions
        
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_squad_hourly_report"

), account as (

    select *
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_account_history"
    where is_most_recent_record = true

), ad_squads as (

    select *
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_squad_history"
    where is_most_recent_record = true

), campaigns as (

    select *
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__campaign_history"
    where is_most_recent_record = true


), aggregated as (

    select
        ad_squad_hourly.source_relation,
        cast(ad_squad_hourly.date_hour as date) as date_day,
        account.ad_account_id,
        account.ad_account_name,
        campaigns.campaign_id,
        campaigns.campaign_name,
        ad_squad_hourly.ad_squad_id,
        ad_squads.ad_squad_name,
        account.currency,
        sum(ad_squad_hourly.swipes) as swipes,
        sum(ad_squad_hourly.impressions) as impressions,
        round(sum(ad_squad_hourly.spend),2) as spend,
        sum(ad_squad_hourly.total_conversions) as total_conversions,
        round(cast(sum(ad_squad_hourly.conversion_purchases_value) as numeric(28,6)), 2) as conversion_purchases_value

        



    
    



    



    
    
    
        
        , sum(coalesce(conversion_purchases, 0)) as conversion_purchases
        

    



        
        





    






    from ad_squad_hourly
    left join ad_squads
        on ad_squad_hourly.ad_squad_id = ad_squads.ad_squad_id
        and ad_squad_hourly.source_relation = ad_squads.source_relation
    left join campaigns
        on ad_squads.campaign_id = campaigns.campaign_id
        and ad_squads.source_relation = campaigns.source_relation
    left join account
        on campaigns.ad_account_id = account.ad_account_id
        and campaigns.source_relation = account.source_relation
    
    group by 1,2,3,4,5,6,7,8,9

)

select *
from aggregated