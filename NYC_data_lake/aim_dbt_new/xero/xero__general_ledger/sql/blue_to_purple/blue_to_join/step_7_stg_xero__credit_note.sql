

with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__credit_note_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as numeric(28,6)) as 
    
    applied_amount
    
 , 
    cast(null as TEXT) as 
    
    branding_theme_id
    
 , 
    
    
    contact_id
    
 as 
    
    contact_id
    
, 
    
    
    credit_note_id
    
 as 
    
    credit_note_id
    
, 
    cast(null as TEXT) as 
    
    credit_note_number
    
 , 
    cast(null as TEXT) as 
    
    currency_code
    
 , 
    cast(null as numeric(28,6)) as 
    
    currency_rate
    
 , 
    cast(null as date) as 
    
    date
    
 , 
    cast(null as date) as 
    
    due_date
    
 , 
    cast(null as date) as 
    
    fully_paid_on_date
    
 , 
    cast(null as boolean) as 
    
    has_attachments
    
 , 
    cast(null as TEXT) as 
    
    line_amount_types
    
 , 
    cast(null as TEXT) as 
    
    reference
    
 , 
    cast(null as numeric(28,6)) as 
    
    remaining_credit
    
 , 
    cast(null as boolean) as 
    
    sent_to_contact
    
 , 
    cast(null as TEXT) as 
    
    status
    
 , 
    cast(null as numeric(28,6)) as 
    
    sub_total
    
 , 
    cast(null as numeric(28,6)) as 
    
    total
    
 , 
    cast(null as numeric(28,6)) as 
    
    total_tax
    
 , 
    cast(null as TEXT) as 
    
    type
    
 , 
    cast(null as timestamp) as 
    
    updated_date_utc
    
 



        



    
    from base
),

final as (
    
    select 
        credit_note_id,
        contact_id

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * from final