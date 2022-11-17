import datetime

set_time = 0;

def speaking(client):
    @client.event
    async def on_message(message):
        print(client.user)
        if message.author == client.user:
            return
        time_obj = int(str(datetime.datetime.now())[11:19].replace(':', ""))
        global set_time
        print(time_obj - set_time)
        if time_obj - set_time < 100:
            await message.channel.send('Blablabla')
        elif message.content.lower().startswith('kentin')\
            or message.content.startswith("<@1042421290407047168>"):
            set_time = time_obj;
            if (len(message.content) == 6 or message.content == ("<@1042421290407047168>")):
                await message.channel.send('Quoi ?')
            else:
                await message.channel.send('Tu veux que je te peel avec ma lulu ?')
        pass_true = 1;