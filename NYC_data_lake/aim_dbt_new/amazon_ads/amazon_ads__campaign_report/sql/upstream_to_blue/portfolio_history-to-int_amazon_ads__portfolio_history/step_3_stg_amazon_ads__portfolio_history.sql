

with base as (

    select * 
    from "amazon_ads"."public_amazon_ads_dev"."stg_amazon_ads__portfolio_history_tmp"
),

fields as (

    select
        
    
    
    budget_amount
    
 as 
    
    budget_amount
    
, 
    
    
    budget_currency_code
    
 as 
    
    budget_currency_code
    
, 
    
    
    budget_end_date
    
 as 
    
    budget_end_date
    
, 
    
    
    budget_policy
    
 as 
    
    budget_policy
    
, 
    
    
    budget_start_date
    
 as 
    
    budget_start_date
    
, 
    
    
    creation_date
    
 as 
    
    creation_date
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    in_budget
    
 as 
    
    in_budget
    
, 
    
    
    last_updated_date
    
 as 
    
    last_updated_date
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    profile_id
    
 as 
    
    profile_id
    
, 
    
    
    serving_status
    
 as 
    
    serving_status
    
, 
    
    
    state
    
 as 
    
    state
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        cast(id as TEXT) as portfolio_id,
        budget_amount,
        budget_currency_code,
        budget_end_date,
        budget_policy,
        budget_start_date,
        creation_date,
        in_budget,
        last_updated_date,
        name as portfolio_name,
        cast(profile_id as TEXT) as profile_id,
        serving_status,
        state,
        row_number() over (partition by source_relation, id order by last_updated_date desc) = 1 as is_most_recent_record
    from fields
)

select *
from final