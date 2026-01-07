with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__application_history_tmp"

    where not coalesce(is_deleted, false)

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    agent_policy_prod
    
 as 
    
    agent_policy_prod
    
, 
    
    
    agent_policy_staging
    
 as 
    
    agent_policy_staging
    
, 
    
    
    agent_version_prod
    
 as 
    
    agent_version_prod
    
, 
    
    
    agent_version_staging
    
 as 
    
    agent_version_staging
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    created_by_user_id
    
 as 
    
    created_by_user_id
    
, 
    
    
    description
    
 as 
    
    description
    
, 
    
    
    disabled_at
    
 as 
    
    disabled_at
    
, 
    
    
    display_name
    
 as 
    
    display_name
    
, 
    
    
    event_count
    
 as 
    
    event_count
    
, 
    
    
    event_rate
    
 as 
    
    event_rate
    
, 
    
    
    first_event_time
    
 as 
    
    first_event_time
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    integrated
    
 as 
    
    integrated
    
, 
    
    
    is_deleted
    
 as 
    
    is_deleted
    
, 
    
    
    last_updated_at
    
 as 
    
    last_updated_at
    
, 
    
    
    last_updated_by_user_id
    
 as 
    
    last_updated_by_user_id
    
, 
    
    
    marked_for_deletion_at
    
 as 
    
    marked_for_deletion_at
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    platform
    
 as 
    
    platform
    
, 
    
    
    push_application_id
    
 as 
    
    push_application_id
    
, 
    
    
    record_until
    
 as 
    
    record_until
    
, 
    
    
    sampling_rate
    
 as 
    
    sampling_rate
    
, 
    
    
    starting_event_time
    
 as 
    
    starting_event_time
    
, 
    
    
    subscription_id
    
 as 
    
    subscription_id
    



        
    from base
),

final as (
    
    select 
        id as application_id,
        agent_policy_prod,
        agent_policy_staging,
        agent_version_prod,
        agent_version_staging,
        created_at,
        created_by_user_id,
        description,
        display_name,
        event_count,
        first_event_time as first_event_at,
        integrated as is_integrated,
        is_deleted,
        last_updated_at,
        last_updated_by_user_id,
        name as application_name,
        platform,
        subscription_id,
        _fivetran_synced

    from fields
)

select * 
from final