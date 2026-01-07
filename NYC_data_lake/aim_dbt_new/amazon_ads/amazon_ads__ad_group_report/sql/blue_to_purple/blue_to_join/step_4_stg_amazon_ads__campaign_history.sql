

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__campaign_history_tmp"
),

fields as (

    select
        
    
    
    bidding_strategy
    
 as 
    
    bidding_strategy
    
, 
    
    
    creation_date
    
 as 
    
    creation_date
    
, 
    
    
    end_date
    
 as 
    
    end_date
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    last_updated_date
    
 as 
    
    last_updated_date
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    portfolio_id
    
 as 
    
    portfolio_id
    
, 
    
    
    profile_id
    
 as 
    
    profile_id
    
, 
    
    
    serving_status
    
 as 
    
    serving_status
    
, 
    
    
    start_date
    
 as 
    
    start_date
    
, 
    
    
    state
    
 as 
    
    state
    
, 
    
    
    targeting_type
    
 as 
    
    targeting_type
    
, 
    
    
    budget
    
 as 
    
    budget
    
, 
    
    
    budget_type
    
 as 
    
    budget_type
    
, 
    
    
    effective_budget
    
 as 
    
    effective_budget
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(id as TEXT) as campaign_id,
        last_updated_date,
        bidding_strategy,
        creation_date,
        end_date,
        name as campaign_name,
        cast(portfolio_id as TEXT) as portfolio_id,
        cast(profile_id as TEXT) as profile_id,
        serving_status,
        start_date,
        state,
        targeting_type,
        budget,
        budget_type,
        effective_budget,
        row_number() over (partition by source_relation, id order by last_updated_date desc) = 1 as is_most_recent_record
    from fields
)

select *
from final