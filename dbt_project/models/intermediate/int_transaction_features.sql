-- models/intermediate/int_transaction_features.sql
-- PURPOSE: Engineer new features from cleaned data
-- These features will be used by both the ML model and Power BI
-- Feature engineering is what separates good analysts from great ones

with stg as (

    -- Always reference other dbt models using ref()
    -- Never hardcode table names -- this is dbt best practice
    select * from {{ ref('stg_transactions') }}

),

features as (

    select
        -- All original columns
        transaction_hour,
        transaction_type,
        amount,
        sender_id,
        sender_balance_before,
        sender_balance_after,
        receiver_id,
        receiver_balance_before,
        receiver_balance_after,
        is_fraud,
        is_flagged_fraud,

        -- FEATURE 1: Sender balance difference
        -- Did the sender's balance drop by exactly the transaction amount?
        -- Fraudsters often drain accounts completely
        round(sender_balance_before - sender_balance_after, 2) as sender_balance_diff,

        -- FEATURE 2: Did sender balance drop to zero?
        -- A balance dropping to exactly zero is highly suspicious
        case
            when sender_balance_after = 0 then 1
            else 0
        end as sender_balance_zero,

        -- FEATURE 3: Receiver balance difference
        round(receiver_balance_after - receiver_balance_before, 2) as receiver_balance_diff,

        -- FEATURE 4: Is the receiver a merchant?
        -- Merchants start with 'M' in this dataset
        case
            when receiver_id like 'M%' then 1
            else 0
        end as receiver_is_merchant,

        -- FEATURE 5: Transaction amount category
        -- Bucket amounts into meaningful business categories
        case
            when amount < 1000    then 'Small'
            when amount < 10000   then 'Medium'
            when amount < 100000  then 'Large'
            else 'Very Large'
        end as amount_category,

        -- FEATURE 6: Time of day
        -- Fraudulent transactions cluster at certain hours
        case
            when transaction_hour % 24 between 0 and 5   then 'Night'
            when transaction_hour % 24 between 6 and 11  then 'Morning'
            when transaction_hour % 24 between 12 and 17 then 'Afternoon'
            else 'Evening'
        end as time_of_day,

        -- FEATURE 7: Balance mismatch flag
        -- If money left sender but didnt arrive at receiver thats suspicious
        case
            when round(sender_balance_before - sender_balance_after, 2) !=
                 round(receiver_balance_after - receiver_balance_before, 2)
            then 1
            else 0
        end as balance_mismatch

    from stg

)

select * from features