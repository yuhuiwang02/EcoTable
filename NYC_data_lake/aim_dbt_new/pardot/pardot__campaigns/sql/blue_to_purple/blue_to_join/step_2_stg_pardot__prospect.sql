with base as (

    select * 
    from "pardot"."public_stg_pardot"."stg_pardot__prospect_tmp"

),

fields as (

    select
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    address_one
    
 as 
    
    address_one
    
, 
    
    
    address_two
    
 as 
    
    address_two
    
, 
    
    
    annual_revenue
    
 as 
    
    annual_revenue
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    city
    
 as 
    
    city
    
, 
    
    
    comments
    
 as 
    
    comments
    
, 
    
    
    company
    
 as 
    
    company
    
, 
    
    
    country
    
 as 
    
    country
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    crm_account_fid
    
 as 
    
    crm_account_fid
    
, 
    
    
    crm_contact_fid
    
 as 
    
    crm_contact_fid
    
, 
    
    
    crm_last_sync
    
 as 
    
    crm_last_sync
    
, 
    
    
    crm_lead_fid
    
 as 
    
    crm_lead_fid
    
, 
    
    
    crm_owner_fid
    
 as 
    
    crm_owner_fid
    
, 
    
    
    crm_url
    
 as 
    
    crm_url
    
, 
    
    
    department
    
 as 
    
    department
    
, 
    
    
    email
    
 as 
    
    email
    
, 
    
    
    employees
    
 as 
    
    employees
    
, 
    
    
    fax
    
 as 
    
    fax
    
, 
    
    
    first_name
    
 as 
    
    first_name
    
, 
    
    
    grade
    
 as 
    
    grade
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    industry
    
 as 
    
    industry
    
, 
    
    
    is_do_not_call
    
 as 
    
    is_do_not_call
    
, 
    
    
    is_do_not_email
    
 as 
    
    is_do_not_email
    
, 
    
    
    is_reviewed
    
 as 
    
    is_reviewed
    
, 
    
    
    is_starred
    
 as 
    
    is_starred
    
, 
    
    
    job_title
    
 as 
    
    job_title
    
, 
    
    
    last_activity_at
    
 as 
    
    last_activity_at
    
, 
    
    
    last_name
    
 as 
    
    last_name
    
, 
    
    
    notes
    
 as 
    
    notes
    
, 
    
    
    opted_out
    
 as 
    
    opted_out
    
, 
    
    
    password
    
 as 
    
    password
    
, 
    
    
    phone
    
 as 
    
    phone
    
, 
    
    
    prospect_account_id
    
 as 
    
    prospect_account_id
    
, 
    
    
    recent_interaction
    
 as 
    
    recent_interaction
    
, 
    
    
    salutation
    
 as 
    
    salutation
    
, 
    
    
    score
    
 as 
    
    score
    
, 
    
    
    source
    
 as 
    
    source
    
, 
    
    
    state
    
 as 
    
    state
    
, 
    
    
    territory
    
 as 
    
    territory
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    
, 
    
    
    user_id
    
 as 
    
    user_id
    
, 
    
    
    website
    
 as 
    
    website
    
, 
    
    
    years_in_business
    
 as 
    
    years_in_business
    
, 
    
    
    zip
    
 as 
    
    zip
    




        
        
    from base
    where not coalesce(_fivetran_deleted, false)
),

final as (
    
    select 
        id as prospect_id,
        _fivetran_deleted,
        _fivetran_synced,
        address_one,
        address_two,
        annual_revenue,
        campaign_id,
        city,
        comments,
        company,
        country,
        created_at as created_timestamp,
        crm_account_fid,
        crm_contact_fid,
        crm_last_sync,
        crm_lead_fid,
        crm_owner_fid,
        crm_url,
        department,
        email,
        employees,
        fax,
        first_name,
        grade,
        industry,
        is_do_not_call,
        is_do_not_email,
        is_reviewed,
        is_starred,
        job_title,
        last_activity_at,
        last_name,
        notes,
        opted_out,
        password,
        phone as phone_number,
        prospect_account_id,
        recent_interaction,
        salutation,
        score,
        source as prospect_source,
        state,
        territory,
        updated_at as updated_timestamp,
        user_id,
        website,
        years_in_business,
        zip
        
        
    from fields
)

select * from final