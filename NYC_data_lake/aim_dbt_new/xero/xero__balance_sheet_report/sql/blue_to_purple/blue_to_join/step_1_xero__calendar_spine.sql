with spine as (

    





with rawdata as (

    

    

    with p as (
        select 0 as generated_number union all select 1
    ), unioned as (

    select

    
    p0.generated_number * power(2, 0)
     + 
    
    p1.generated_number * power(2, 1)
     + 
    
    p2.generated_number * power(2, 2)
     + 
    
    p3.generated_number * power(2, 3)
     + 
    
    p4.generated_number * power(2, 4)
     + 
    
    p5.generated_number * power(2, 5)
     + 
    
    p6.generated_number * power(2, 6)
    
    
    + 1
    as generated_number

    from

    
    p as p0
     cross join 
    
    p as p1
     cross join 
    
    p as p2
     cross join 
    
    p as p3
     cross join 
    
    p as p4
     cross join 
    
    p as p5
     cross join 
    
    p as p6
    
    

    )

    select *
    from unioned
    where generated_number <= 84.0
    order by generated_number



),

all_periods as (

    select (
        

    cast('2019-01-01' as date) + ((interval '1 month') * (row_number() over (order by generated_number) - 1))


    ) as date_month
    from rawdata

),

filtered as (

    select *
    from all_periods
    where date_month <= 

    current_date + ((interval '1 month') * (1))



)

select * from filtered



), cleaned as (

    select cast(date_month as date) as date_month
    from spine

)

select *
from cleaned