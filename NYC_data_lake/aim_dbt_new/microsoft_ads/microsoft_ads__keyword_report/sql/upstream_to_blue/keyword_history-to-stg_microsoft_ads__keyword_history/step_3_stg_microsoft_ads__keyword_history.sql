

with base as (

    select * 
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__keyword_history_tmp"
),

fields as (

    select
        
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    modified_time
    
 as 
    
    modified_time
    
, 
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    match_type
    
 as 
    
    match_type
    
, 
    
    
    status
    
 as 
    
    status
    



        
    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        id as keyword_id,
        name as keyword_name,
        modified_time as modified_at,
        ad_group_id,
        match_type,
        status,
        row_number() over (partition by source_relation, id order by modified_time desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final