with calendar as (

    select *
    from "xero"."public_xero_dev"."xero__calendar_spine"

), ledger as (

    select *
    from "xero"."public_xero_dev"."xero__general_ledger"

), organization as (

    select 
        *,
        cast(extract(year from current_date) as TEXT) as current_year,
        cast(extract(year from 

    current_date + ((interval '1 year') * (1))

) as TEXT) as next_year
    from "xero"."public_xero_dev"."stg_xero__organization"


), year_end as (

-- Calculate the current financial year-end date for each organization:
-- For February, determine last day by subtracting 1 day from March 1, avoiding leap year logic.
-- Compare the year end date to the current date:
--   Use this year's date if it's on or after the current date.
--   Otherwise, use the next year's corresponding date.
    select 
        source_relation,
        case when financial_year_end_month = 2 and financial_year_end_day = 29
            then
                case when cast(

    cast(current_year || '-03-01' as date) + ((interval '1 day') * (-1))

 as date) >= current_date
                    then cast(

    cast(current_year || '-03-01' as date) + ((interval '1 day') * (-1))

 as date)
                    else cast(

    cast(next_year || '-03-01' as date) + ((interval '1 day') * (-1))

 as date)
                    end
            else
                case when cast(current_year || '-' || financial_year_end_month || '-' || financial_year_end_day as date) >= current_date
                    then cast(current_year || '-' || financial_year_end_month || '-' || financial_year_end_day as date)
                    else cast(next_year || '-' || financial_year_end_month || '-' || financial_year_end_day as date)
                    end
        end as current_year_end_date

    from organization

), joined as (

    select
        calendar.date_month,
        case
            when ledger.account_class in ('ASSET','EQUITY','LIABILITY') then ledger.account_name
            when ledger.journal_date <= 

    year_end.current_year_end_date + ((interval '1 year') * (-1))

 then 'Retained Earnings'
            else 'Current Year Earnings'
        end as account_name,
        case
            when ledger.account_class in ('ASSET','EQUITY','LIABILITY') then ledger.account_code
            else null
        end as account_code,
        case
            when ledger.account_class in ('ASSET','EQUITY','LIABILITY') then ledger.account_id
            else null
        end as account_id,
        case
            when ledger.account_class in ('ASSET','EQUITY','LIABILITY') then ledger.account_type
            else null
        end as account_type,
        case
            when ledger.account_class in ('ASSET','EQUITY','LIABILITY') then ledger.account_class
            else 'EQUITY'
        end as account_class,
        ledger.source_relation, 
        sum(ledger.net_amount) as net_amount
    from calendar
    inner join ledger
        on calendar.date_month >= cast(date_trunc('month', ledger.journal_date) as date)
    cross join year_end
	where year_end.source_relation = ledger.source_relation
    group by 1,2,3,4,5,6,7

)

select *
from joined