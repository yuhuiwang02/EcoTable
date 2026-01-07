
with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__ad_group_level_report_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    campaign_bidding_strategy
    
 as 
    
    campaign_bidding_strategy
    
, 
    
    
    clicks
    
 as 
    
    clicks
    
, 
    
    
    cost
    
 as 
    
    cost
    
, 
    
    
    date
    
 as 
    
    date
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    cast(null as integer) as 
    
    purchases_30_d
    
 , 
    cast(null as float) as 
    
    sales_30_d
    
 


    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(ad_group_id as TEXT) as ad_group_id,
        campaign_bidding_strategy,
        clicks,
        cost,
        date as date_day,
        impressions,
        purchases_30_d,
        sales_30_d
        
        





    from fields
)

select *
from final