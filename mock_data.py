import random

PROJECTS = [
    "BookStack", "Boostnote", "FFmpeg", "Files", "Ghost", "ILSpy", "Jackett",
    "Lychee", "MuseScore", "NodeBB", "Ombi", "QGIS", "Radarr", "Rocket.Chat",
    "ShareX", "Sonarr", "WordPress", "ansible", "app", "bitcoin", "brew",
    "caddy", "calibre", "cassandra", "cli", "core", "curl", "desktop",
    "diaspora", "directus", "discourse", "dnSpy", "drawio", "elasticsearch",
    "etherpad-lite", "excalidraw", "flarum", "forem", "freeCodeCamp",
    "geoserver", "git", "gitea", "gitlabhq", "godot", "graylog2-server",
    "hadoop", "hyper", "immich", "influxdb", "jekyll", "jellyfin", "jenkins",
    "jitsi-meet", "joplin", "keycloak", "krita", "kubernetes", "lens",
    "linux", "loki", "mastodon", "matomo", "minio", "moodle", "neovim",
    "nginx", "nifi", "node-red", "notepad-plus-plus", "obs-studio", "odoo",
    "openpilot", "openproject", "openttd", "osTicket", "outline", "panel",
    "paperless-ngx", "phpmyadmin", "prometheus", "qBittorrent", "redis",
    "redmine", "salt", "sentry", "server", "solr", "spree", "supabase",
    "systemd", "tdesktop", "terraform", "tmux", "traefik", "trino", "vim",
    "vscode", "whisper", "yt-dlp"
]

AUTHORS = [
    "Ahmed Nasser", "Alex Gaillard", "Ava Chow", "Barış Soner Uşaklı",
    "Ben Halpern", "Benoît Viguier", "Björn Rabenstein", "Casper Jeukendrup",
    "Chocobo1", "Chris Hostetter", "Chris Raible", "Dan Brown",
    "Daniel Dietzler", "Daniel Wozniak", "David Benson", "David Luzar",
    "Denis Peshkov", "Dennis Schubert", "Don HO", "Drake Costa",
    "Florent de Labarre", "Foxe Chen", "Garfield69", "GitLab Bot",
    "Harrissou Sant-anna", "J. Nick Koston", "Jaex",
    "Jalil David Salamé Messina", "Jamie Strandboge", "Jaya Allamsetty",
    "JediKev", "John Molakvoæ", "John Preston", "Jong Wook Kim",
    "Junio C Hamano", "KZ", "Kacper Michajłow", "Ketan Kelkar",
    "Kolin Newby", "Kovid Goyal", "Kris", "Kubernetes Prow Robot",
    "Kévin Dunglas", "Lars Francke", "Laurent Cozic", "Linus Torvalds",
    "Luca Boccassi", "Marius Balteanu", "Mark McDowall", "Matt Jankowski",
    "Matthew Penner", "Maurício Meneghini Fauth", "Megan Rogge",
    "Mike McQuaid", "Nathan Gavin", "Nick O'Leary", "Paulo Motta",
    "Pierre Villard", "Rob Howell", "Sami Mazouz", "Sara Arjona",
    "Sarah French", "Sebastiaan van Stijn", "Sergey Biryukov",
    "Sergey Kandaurov", "Shayna Chambless", "Siegfried Pammer",
    "Steven Hawkins", "Thaddeus Crews", "Thomas Adam", "Todd Martin",
    "Tom Moor", "Trenton H", "Tyler Trahan", "Viktor Szakats",
    "Wolthera van Hövell tot Westerflier", "Yago Raña Gayoso", "Yair",
    "bashonly", "brozek", "dependabot[bot]", "eFini",
    "elasticsearchmachine", "fbaldo31", "jekyllbot", "kevinpollet",
    "mengxun", "peem5210", "renovate[bot]", "techknowlogick", "tidy-dev",
    "translatewiki.net", "zhtttylz"
]

QUALITIES = ["good", "moderate", "poor"]

def mock_predictions(target):

    if target == "project":
        names = random.sample(PROJECTS, 3)
    elif target == "author":
        names = random.sample(AUTHORS, 3)
    elif target == "quality":
        names = random.sample(QUALITIES, 3)

    valores = [random.random() for _ in range(3)]
    total = sum(valores)
    confiancas = [round(v / total, 4) for v in valores]

    return {
        "target": target,
        "predictions": [
            {"class": names[0], "confidence": confiancas[0]},
            {"class": names[1], "confidence": confiancas[1]},
            {"class": names[2], "confidence": confiancas[2]}
        ]
    }