with base as (

    select * 
    from "pardot"."public_stg_pardot"."stg_pardot__opportunity_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    closed_at
    
 as 
    
    closed_at
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    probability
    
 as 
    
    probability
    
, 
    
    
    stage
    
 as 
    
    stage
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    type
    
 as 
    
    type
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    
, 
    
    
    value
    
 as 
    
    value
    



        
    from base
),

final as (
    
    select 
        id as opportunity_id,
        campaign_id,
        created_at as created_timestamp,
        updated_at as updated_timestamp,
        name as opportunity_name,
        probability,
        status as opportunity_status,
        stage,
        type as opportunity_type,
        value as amount,
        _fivetran_synced,
        closed_at as closed_timestamp
    from fields

)

select * from final