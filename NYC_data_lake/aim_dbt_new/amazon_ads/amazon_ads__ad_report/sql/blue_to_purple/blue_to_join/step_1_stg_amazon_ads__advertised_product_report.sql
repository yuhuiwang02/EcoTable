

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__advertised_product_report_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    ad_id
    
 as 
    
    ad_id
    
, 
    
    
    advertised_asin
    
 as 
    
    advertised_asin
    
, 
    
    
    advertised_sku
    
 as 
    
    advertised_sku
    
, 
    
    
    campaign_budget_amount
    
 as 
    
    campaign_budget_amount
    
, 
    
    
    campaign_budget_currency_code
    
 as 
    
    campaign_budget_currency_code
    
, 
    
    
    campaign_budget_type
    
 as 
    
    campaign_budget_type
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
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
        cast(ad_id as TEXT) as ad_id,
        cast(ad_group_id as TEXT) as ad_group_id,
        advertised_asin,
        advertised_sku,
        campaign_budget_amount,
        campaign_budget_currency_code,
        campaign_budget_type,
        cast(campaign_id as TEXT) as campaign_id,
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