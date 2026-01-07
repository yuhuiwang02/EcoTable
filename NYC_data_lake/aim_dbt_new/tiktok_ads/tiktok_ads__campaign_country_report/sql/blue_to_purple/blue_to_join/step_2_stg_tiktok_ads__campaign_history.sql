

with base as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__campaign_history_tmp"
), 

fields as (

    select
        
    
    
    advertiser_id
    
 as 
    
    advertiser_id
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    campaign_name
    
 as 
    
    campaign_name
    
, 
    
    
    campaign_type
    
 as 
    
    campaign_type
    
, 
    
    
    split_test_variable
    
 as 
    
    split_test_variable
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    
, 
    
    
    objective_type
    
 as 
    
    objective_type
    
, 
    cast(null as TEXT) as 
    
    status
    
 , 
    
    
    budget
    
 as 
    
    budget
    
, 
    
    
    budget_mode
    
 as 
    
    budget_mode
    
, 
    
    
    create_time
    
 as 
    
    create_time
    
, 
    
    
    is_new_structure
    
 as 
    
    is_new_structure
    



    
        


, cast('' as TEXT) as source_relation




    from base
), 

final as (

    select
        source_relation,   
        campaign_id,
        cast(updated_at as timestamp) as updated_at,
        advertiser_id,
        campaign_name,
        campaign_type,
        split_test_variable,
        objective_type,
        status,
        budget,
        budget_mode,
        create_time as created_at,
        is_new_structure,
        row_number() over (partition by source_relation, campaign_id order by updated_at desc) = 1 as is_most_recent_record
    from fields
)

select *
from final