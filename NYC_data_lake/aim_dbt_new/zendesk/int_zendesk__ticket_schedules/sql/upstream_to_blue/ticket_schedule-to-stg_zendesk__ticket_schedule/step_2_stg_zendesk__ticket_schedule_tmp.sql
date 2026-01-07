--To disable this model, set the using_schedules variable within your dbt_project.yml file to False.









    select
            "created_at",
  "ticket_id",
  "_fivetran_synced",
  "schedule_id"
        from "google_ads"."public"."ticket_schedule_data" as source_table
    
    