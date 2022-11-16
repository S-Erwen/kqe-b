def speaking(client):
    @client.event
    async def on_message(message):
        print(client.user)
        if message.author == client.user:
            return
        if message.content.lower().startswith('kentin')\
            or message.content.startswith("<@1042421290407047168>"):
            if (len(message.content) == 6):
                await message.channel.send('Quoi ?')
            else:
                await message.channel.send('Tu veux que je te peel avec ma lulu ?')