import pygame
import pandas as pd
import time
import math
import sys


print('Reading emissions data')

emissions_path = sys.argv[1]
df = pd.read_csv(emissions_path)
#  time,step,id,position,speed,accel,headway,leader_speed,speed_difference,leader_id,follower_id,instant_energy_consumption,total_energy_consumption,total_distance_traveled,total_miles,total_gallons,avg_mpg

timesteps = sorted(list(set(map(lambda x: round(x, 1), df['time']))))

car_types = []
car_positions = []
car_speeds = []

for ts in timesteps:
    ts_data = df.loc[df['time'] == ts]
    car_types.append(list(ts_data['id']))
    car_positions.append(list(ts_data['position']))
    car_speeds.append(list(ts_data['speed']))

print('Done')

pygame.init()
screen = pygame.display.set_mode([1500, 500])



screen_rect = screen.get_rect()
mid_x = screen_rect.centerx
mid_y = screen_rect.centery
zoom = 5
timestep = 0.1
interval = 100

font = pygame.font.SysFont('Verdana', 10)

pos_x = 0

running = True
t0 = time.time() - timesteps[0]
for i in range(len(timesteps)):
    t = timesteps[i]
    if timestep < .05:
        time.sleep(max(t0 + t - time.time(), 0))
    else:
        t0 = time.time() - t
        if int(t * 10) % int(timestep * 10) != 0:
            continue
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN: # or event.type == pygame.KEYUP:
            if event.mod == pygame.KMOD_NONE:
                if event.key == 1073741903: timestep += 0.1  # right
                elif event.key == 1073741904: timestep -= 0.1  # left
                timestep = min(max(timestep, 0.0), 2.0)
            elif event.mod & pygame.KMOD_SHIFT:
                if event.key == 1073741903: zoom += 1  # right
                elif event.key == 1073741904: zoom -= 1  # left
                zoom = min(max(zoom, 1), 10)
            elif event.mod & pygame.KMOD_CTRL:
                if event.key == 1073741903: interval += 10  # right
                elif event.key == 1073741904: interval -= 10  # left
                interval = min(max(interval, 20), 1000)

    screen.fill((255, 255, 255))

    img = font.render(f'Timestep: {round(timestep, 1)}s (commands: Left / Right) (0 = real time)', True, (0, 0, 0))
    screen.blit(img, (20, 50))
    img = font.render(f'Zoom: x{int(zoom)} (commands: Shift + Left / Right)', True, (0, 0, 0))
    screen.blit(img, (20, 70))
    img = font.render(f'Interval: {int(interval)}m (commands: Ctrl + Left / Right)', True, (0, 0, 0))
    screen.blit(img, (20, 90))

    img = font.render(f'Time: {round(t, 1)}s', True, (0, 0, 0))
    screen.blit(img, (20, 130))
    img = font.render(f'Distance: {round(max(car_positions[i]), 1)}m', True, (0, 0, 0))
    screen.blit(img, (20, 150))

    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, mid_y - 10 * zoom, screen_rect.width, 2 * 10 * zoom))

    shift_x = 0
    for car_type, car_x in zip(car_types[i], car_positions[i]):
        if car_type.startswith('leader'):
            shift_x = mid_x - car_x * zoom

    for pos_x in range(-1000, 22000, interval):
        img = font.render(f'{interval}m', True, (0, 100, 0))
        screen.blit(img, ((pos_x + interval // 2) * zoom + shift_x - img.get_width() // 2, 10 - img.get_height() // 2))
        pygame.draw.rect(screen, (0, 200, 0), pygame.Rect(pos_x * zoom - 1 + shift_x, 0, 2, 20))

        img = font.render(f'{interval}m', True, (0, 100, 0))
        screen.blit(img, ((pos_x + interval // 2) * zoom + shift_x - img.get_width() // 2, screen_rect.height - 10 - img.get_height() // 2))
        pygame.draw.rect(screen, (0, 200, 0), pygame.Rect(pos_x * zoom - 1 + shift_x, screen_rect.height - 20, 2, 20))

    for car_type, car_x, car_speed in zip(car_types[i], car_positions[i], car_speeds[i]):
        pos_x = car_x * zoom + shift_x
        pos_y = mid_y
        radius = 3 * zoom

        if 'leader' in car_type:
            car_color = (0, 255, 0)
        elif 'human' in car_type:
            car_color = (255, 255, 255)
        elif 'av' in car_type:
            car_color = (255, 0, 0)

        pygame.draw.circle(screen, car_color, (pos_x, pos_y), radius)

        img = font.render(car_type, True, (0, 0, 0))
        screen.blit(img, (pos_x - img.get_width() // 2, pos_y - 20 * zoom - img.get_height() // 2))

        img = font.render(f'{round(car_speed, 1)}' + (' m/s' if car_type.startswith('leader') else ''), True, (0, 0, 0))
        screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * zoom - img.get_height() // 2))
        img = font.render(f'{round(car_speed * 3.6, 1)}' + (' km/h' if car_type.startswith('leader') else ''), True, (0, 0, 0))
        screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * zoom + 15 - img.get_height() // 2))
        img = font.render(f'{round(car_speed * 2.237, 1)}' + (' mph' if car_type.startswith('leader') else ''), True, (0, 0, 0))
        screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * zoom + 30 - img.get_height() // 2))

    pygame.display.flip()

    if not running:
        break

pygame.quit()