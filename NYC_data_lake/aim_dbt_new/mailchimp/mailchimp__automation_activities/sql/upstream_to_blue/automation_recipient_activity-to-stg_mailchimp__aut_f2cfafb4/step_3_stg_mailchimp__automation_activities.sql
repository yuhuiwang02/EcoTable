

with base as (

    select *
    from "mailchimp"."public_mailchimp_dev"."stg_mailchimp__automation_activities_tmp"

), 

fields as (

    select 
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    action
    
 as 
    
    action
    
, 
    
    
    automation_email_id
    
 as 
    
    automation_email_id
    
, 
    
    
    bounce_type
    
 as 
    
    bounce_type
    
, 
    
    
    ip
    
 as 
    
    ip
    
, 
    
    
    list_id
    
 as 
    
    list_id
    
, 
    
    
    member_id
    
 as 
    
    member_id
    
, 
    
    
    timestamp
    
 as 
    
    timestamp
    
, 
    
    
    url
    
 as 
    
    url
    



        
    from base

), 

final as (

    select 
        action as action_type,
        automation_email_id,
        member_id,
        list_id,
        timestamp as activity_timestamp,
        ip as ip_address,
        url,
        bounce_type
    from fields

),

unique_key as (

    select 
        *, 
        md5(cast(coalesce(cast(action_type as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(automation_email_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(member_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(activity_timestamp as TEXT), '_dbt_utils_surrogate_key_null_') as TEXT)) as activity_id
    from final
)

select * from unique_key