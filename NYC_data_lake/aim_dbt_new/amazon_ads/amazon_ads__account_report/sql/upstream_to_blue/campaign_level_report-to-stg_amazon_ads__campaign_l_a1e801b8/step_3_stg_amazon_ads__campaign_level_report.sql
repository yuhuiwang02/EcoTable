

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__campaign_level_report_tmp"
),

fields as (

    select
        
    
    
    campaign_applicable_budget_rule_id
    
 as 
    
    campaign_applicable_budget_rule_id
    
, 
    
    
    campaign_applicable_budget_rule_name
    
 as 
    
    campaign_applicable_budget_rule_name
    
, 
    
    
    campaign_bidding_strategy
    
 as 
    
    campaign_bidding_strategy
    
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
    
    
    campaign_rule_based_budget_amount
    
 as 
    
    campaign_rule_based_budget_amount
    
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
    
    
    purchases_30_d
    
 as 
    
    purchases_30_d
    
, 
    
    
    sales_30_d
    
 as 
    
    sales_30_d
    
, 
    
    
    sales_7_d
    
 as 
    
    sales_7_d
    
, 
    
    
    purchases_30_d
    
 as purchases_alias , 
    cast(null as TEXT) as 
    
    purchases_14_d
    
 


    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        campaign_applicable_budget_rule_id,
        campaign_applicable_budget_rule_name,
        campaign_bidding_strategy,
        campaign_budget_amount,
        campaign_budget_currency_code,
        campaign_budget_type,
        cast(campaign_id as TEXT) as campaign_id,
        campaign_rule_based_budget_amount,
        clicks,
        cost,
        date as date_day,
        impressions,
        purchases_30_d,
        sales_30_d

        


    
        
            
                , coalesce(cast(sales_7_d as float), 0) as sales_7_d
            
        
    
        
            
                , coalesce(cast(purchases_alias as float), 0) as purchases_alias
            
        
    
        
            
                , coalesce(cast(purchases_14_d as float), 0) as purchases_14_d
            
        
    




    from fields
)

select *
from final