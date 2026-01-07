

with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__bank_transaction_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    bank_account_id
    
 , 
    
    
    bank_transaction_id
    
 as 
    
    bank_transaction_id
    
, 
    cast(null as TEXT) as 
    
    batch_payment_batch_payment_id
    
 , 
    cast(null as timestamp) as 
    
    batch_payment_date
    
 , 
    cast(null as TEXT) as 
    
    batch_payment_id
    
 , 
    cast(null as boolean) as 
    
    batch_payment_is_reconciled
    
 , 
    cast(null as TEXT) as 
    
    batch_payment_status
    
 , 
    cast(null as float) as 
    
    batch_payment_total_amount
    
 , 
    cast(null as TEXT) as 
    
    batch_payment_type
    
 , 
    cast(null as timestamp) as 
    
    batch_payment_updated_date_utc
    
 , 
    
    
    contact_id
    
 as 
    
    contact_id
    
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
    cast(null as TEXT) as 
    
    external_link_provider_name
    
 , 
    cast(null as boolean) as 
    
    has_attachments
    
 , 
    cast(null as boolean) as 
    
    is_reconciled
    
 , 
    cast(null as TEXT) as 
    
    line_amount_types
    
 , 
    cast(null as TEXT) as 
    
    overpayment_id
    
 , 
    cast(null as TEXT) as 
    
    prepayment_id
    
 , 
    cast(null as TEXT) as 
    
    reference
    
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
    
 , 
    cast(null as TEXT) as 
    
    url
    
 



        




    from base
),

final as (
    
    select 
        bank_transaction_id,
        contact_id

        


, cast('' as TEXT) as source_relation




    from fields
)

select * from final