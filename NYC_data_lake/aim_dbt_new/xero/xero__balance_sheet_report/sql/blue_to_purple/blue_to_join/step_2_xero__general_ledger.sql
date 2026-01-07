

    


with journals as (

    select *
    from "xero"."public_xero_dev"."stg_xero__journal"

), journal_lines as (

    select *
    from "xero"."public_xero_dev"."stg_xero__journal_line"

), accounts as (

    select *
    from "xero"."public_xero_dev"."stg_xero__account"

), invoices as (

    select *
    from "xero"."public_xero_dev"."stg_xero__invoice"


), bank_transactions as (

    select *
    from "xero"."public_xero_dev"."stg_xero__bank_transaction"




), credit_notes as (

    select *
    from "xero"."public_xero_dev"."stg_xero__credit_note"


), contacts as (

    select *
    from "xero"."public_xero_dev"."stg_xero__contact"


), pivoted_tracking_categories as (

    select *
    from "xero"."public_xero_dev"."int_xero__journal_line_pivoted_tracking_categories"


), joined as (

    select 
        journals.journal_id,
        journals.created_date_utc,
        journals.journal_date,
        journals.journal_number,
        journals.reference,
        journals.source_id,
        journals.source_type,
        journals.source_relation,
        journal_lines.journal_line_id,
        accounts.account_code,
        accounts.account_id,
        accounts.account_name,
        accounts.account_type,
        journal_lines.description,
        journal_lines.gross_amount,
        journal_lines.net_amount,
        journal_lines.tax_amount,
        journal_lines.tax_name,
        journal_lines.tax_type,
        accounts.account_class,

        case when journals.source_type in ('ACCPAY', 'ACCREC') then journals.source_id end as invoice_id,
        case when journals.source_type in ('CASHREC','CASHPAID') then journals.source_id end as bank_transaction_id,
        case when journals.source_type in ('TRANSFER') then journals.source_id end as bank_transfer_id,
        case when journals.source_type in ('MANJOURNAL') then journals.source_id end as manual_journal_id,
        case when journals.source_type in ('APPREPAYMENT', 'APOVERPAYMENT', 'ACCPAYPAYMENT', 'ACCRECPAYMENT', 'ARCREDITPAYMENT', 'APCREDITPAYMENT') then journals.source_id end as payment_id,
        case when journals.source_type in ('ACCPAYCREDIT','ACCRECCREDIT') then journals.source_id end as credit_note_id

        
        -- Pivoted tracking categories, excluding duplicate columns
        
        , pivoted_tracking_categories.region 
        
        , pivoted_tracking_categories.department 
        
        , pivoted_tracking_categories.location 
        
        , pivoted_tracking_categories.project 
        
        

    from journals
    left join journal_lines
        on journals.journal_id = journal_lines.journal_id
        and journals.source_relation = journal_lines.source_relation
    left join accounts
        on accounts.account_id = journal_lines.account_id
        and accounts.source_relation = journal_lines.source_relation
    
    
    left join pivoted_tracking_categories
        on journal_lines.journal_line_id = pivoted_tracking_categories.journal_line_id
        and journals.journal_id = pivoted_tracking_categories.journal_id
        and journals.source_relation = pivoted_tracking_categories.source_relation
    

), first_contact as (

    select 
        joined.*,
        
        coalesce(
            invoices.contact_id
            
                , bank_transactions.contact_id
            

            
            , credit_notes.contact_id
            
        )
        

        as contact_id
    from joined
    left join invoices 
        on (joined.invoice_id = invoices.invoice_id
        and joined.source_relation = invoices.source_relation)
    
    left join bank_transactions
        on (joined.bank_transaction_id = bank_transactions.bank_transaction_id
        and joined.source_relation = bank_transactions.source_relation)
    

    
    left join credit_notes 
        on (joined.credit_note_id = credit_notes.credit_note_id
        and joined.source_relation = credit_notes.source_relation)
    

), second_contact as (

    select 
        first_contact.*,
        contacts.contact_name
    from first_contact
    left join contacts 
        on (first_contact.contact_id = contacts.contact_id
        and first_contact.source_relation = contacts.source_relation)

)

select *
from second_contact