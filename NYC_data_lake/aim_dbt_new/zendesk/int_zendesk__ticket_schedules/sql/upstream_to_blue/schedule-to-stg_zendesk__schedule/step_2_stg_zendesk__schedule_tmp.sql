--To disable this model, set the using_schedules variable within your dbt_project.yml file to False.









    select
            "end_time",
  "id",
  "start_time",
  "_fivetran_deleted",
  "_fivetran_synced",
  "end_time_utc",
  "name",
  "start_time_utc",
  "time_zone",
  "created_at"
        from "google_ads"."public"."schedule_data" as source_table
    
    