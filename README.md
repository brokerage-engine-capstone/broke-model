# brokerage-engine-data-science-team


## Advice From A Programmer

	- Use Branches
	- Use Pull Requests
	- Use TDD when possible
	- Collaborate directly with your teamates
	- If you don't have enough data, let me know
	- If you need resources, let me know

### I will be hosting the csv's on this repo

	- Please Clone the Repo
	- cp the csv's into your team's public repo for storage
	

### How to clone

	- `git clone git@github.com:bahodge/brokerage-engine-data-science-team.git`
	- `cd into brokerage-engine-data-science-team`

### Move csv's into your repo

	- `cp /path/to/csv /path/to/your/public/repo`

### Formats
	- Money is represented in float
		- $1.00 is represented as 1.0
	- Percentage is represented in float
		- 1% is represented as .01
	- Enums
		- Enums are expressed as strings in all caps
		- `"STANDARD"`

# Explanation of data
There are 3 main parts of the system. Calculation, distribution and accounting. I am providing calculation.

### CSV 1 - `agents_with_transactions.csv`

This set of data contains information about the actual payout for each agent by transaction. There are approximately 16000 data points in this set. 

	- Commission Anniversary -> When the time comes to renegotiate the agent's split rate
	- Commission Schedule Effective Start At -> When the commission schedule starts
	- Commission Schedule Effective End at -> when the commission schedule ends
	- Commission Schedule Strategy -> How the commission schedule calculates
	- Transaction Number -> BT 'year' 'month' 'day' 'transaction_count'
		- Unique to a single account, USE transaction_id
	- Transaction Contracted At -> When the buyers and sellers signed contract to begin transaction
	- Transaction Closed At -> When the transaction was closed
	- Transaction Effective At -> An override for when the transaction actually closed
	- Transaction List Amount -> Set by users (WILL TRY TO PULL FROM LISTING SIDE SYSTEM)
	- Transact Status -> Open, ~DONT WORRY ABOUT THESE~ ,Cda Sent, Complete, or Fell through
	- Earned Side Count -> A strange representation of how much credit an agent gets for a transaction
		- Can be split between agents on a side
	- Earned Volume -> Typically is the same as the sales amount
		- Can be split between agents on a side
	- Transaction Side ->
		- Listing Side -> The agent is representing the seller of the property
		- Selling Side -> The agent is representing the buyer of the property
	- Standard Commission Type -> The regular payout from a transaction
	- Standard Commission GCI amount -> 
		- the 3% || base value the brokerage took as commission
		- Before splitting with agent
	- Standard Commission Agent Net -> How much the agent took home
	- Standard Commission Brokerage Net -> How much the brokerage took
	- Bonus Commission -> Any extra money paid out

# I'm just a programmer
	- If you need more data / want to know how things are related, let me know and I'll try to make it for you
	- I have no life and am happy to answer any questions while i'm awake
	- If you really really need help with understanding the domain, pm / call me. 
	- If you are lost in the sauce, I can help you.
	- No, you don't get banking info.
	- If you need anything form the business, let me know, I will try to provide it. 

# Have fun


# Note

These CSV's contain no personal information about Agent's, Addresses, Banking information, or any encrypted proprietary information
